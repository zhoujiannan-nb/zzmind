#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zzmind-0.5B —— 02 架构定稿的落地实现（对应 DESIGN.md v1.0，497.8M）

设计依据（每个组件的"为什么"都写在 DESIGN.md / 图 5B/5C/5D）：
  - GQA 注意力 16Q/8KV、QK-Norm、RoPE(base 1e6)   → L1 §3.1 / §3.2 / §3.6，图 5C
  - RMSNorm pre-norm（残差走干净恒等）            → L1 §3.3，图 5D
  - SwiGLU FFN（silu(gate)·up → down）            → L1 §3.4，图 5D
  - embed 与 lm_head tie（同一张 tensor）         → L1 §3.5
  - 冻结配置 = 下方 MiniMindConfig 的默认值（1280×24、4032、6400、32k）

与 minimind 底座 API 兼容（MiniMindConfig / MiniMindForCausalLM，
forward / generate 签名一致），后续 03 预训练 / 04 SFT / 06 agent 的训练脚本可直接 import。

自检：python 03_pretrain/model.py
"""
import math
import sys
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class MiniMindConfig(PretrainedConfig):
    """冻结配置：默认值 = DESIGN.md §9 定稿（不传参就是 zzmind-0.5B）。"""
    model_type = "minimind"

    def __init__(self, hidden_size=1280, num_hidden_layers=24, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe                    # 0.5B 全参稠密即可，MoE 字段仅兼容保留
        self.dropout = kwargs.get("dropout", 0.0)          # L1 §6：欠训练阶段不配正则
        self.vocab_size = kwargs.get("vocab_size", 6400)   # C3：01 的 BPE 锁死
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)   # torch SDPA
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)   # Q 头（不打折）
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)    # KV 头（GQA 砍半）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 80
        self.hidden_act = kwargs.get("hidden_act", "silu")                 # SwiGLU
        self.intermediate_size = kwargs.get(                                # L2：ceil(H·π/64)·64 = 4032
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # C4
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)                     # L1 §2：对齐 Qwen3/minimind
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # L1 §5
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32, "beta_slow": 1, "factor": 16,
            "original_max_position_embeddings": 2048, "attention_factor": 1.0, "type": "yarn",
        } if self.inference_rope_scaling else None
        # MoE 兼容字段（use_moe=False 时全部闲置）
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    """L1 §3.1：除以均方根（音量旋钮）+ 可学习逐维缩放（均衡器）；无 bias。"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)   # fp32 算、转回原 dtype


def precompute_freqs_cis(dim, end=int(32 * 1024), rope_base=1e6, rope_scaling=None):
    """L1 §2 RoPE：θᵢ = base^(-2i/dim)，base 1e6 → 最慢档周期 ~440 万，32k 覆盖充足。
    返回 cos/sin 表 [end, dim]；可选的 YaRN（rope_scaling）留给未来长上下文外推。"""
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:  # YaRN: 对低频段频率做线性缩放
        orig_max, factor, beta_fast, beta_slow = (rope_scaling.get("original_max_position_embeddings", 2048),
                                                  rope_scaling.get("factor", 16), rope_scaling.get("beta_fast", 32),
                                                  rope_scaling.get("beta_slow", 1))
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
        freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1), torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把位置旋进 q/k（V 不旋转）：旋转矩阵正交 → 点积只剩相对位置差（钟针夹角）。"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x, n_rep):
    """GQA 机关（图 5C）：KV 头 8 → 16（每套复制 2 份，2 个 Q 头共享 1 套 K+V）。
    复制的是数值（expand+reshape，不花新参数），W_k/W_v 不动。"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """图 5C 的落地：三路并行投影 → QK-Norm → RoPE → repeat_kv 对齐 →
    打分(q·kᵀ/√d) + 因果 mask → softmax → 加权取 v → 16 头拼平 → o_proj → 残差。"""
    def __init__(self, config):
        super().__init__()
        self.n_local_heads = config.num_attention_heads      # 16
        self.n_local_kv_heads = config.num_key_value_heads   # 8
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 2（每组共享）
        self.head_dim = config.head_dim                      # 80
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)  # 1280→1280
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)         # 1280→640
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)         # 1280→640
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)    # 1280→1280
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)   # QK-Norm（§3.6，RoPE 之前）
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)      # 三路并行（不是先后）
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)    # 后拆 16×80
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 8×80
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)                   # RoPE 只进 q/k
        if past_key_value is not None:                                    # 推理：历史 K/V 追加进缓存
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2),
                      repeat_kv(xk, self.n_rep).transpose(1, 2),          # GQA: 8→16（图 5C）
                      repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)   # 逐头点积打分（对 80 求和，16 不动）
            if self.is_causal:                                                # 因果 mask：只看过去
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv  # softmax → 权重×v（N 被约掉）
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)            # 16 头拼平 → 1280
        output = self.resid_dropout(self.o_proj(output))                     # o_proj 跨头混合
        return output, past_kv


class FeedForward(nn.Module):
    """图 5D：SwiGLU —— silu(gate)·up → down。gate 是开关（0~1），up 是内容。"""
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size    # 4032
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]                              # silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniMindBlock(nn.Module):
    """图 5D：pre-norm 残差结构 —— 残差走干净恒等，norm 在子层入口。"""
    def __init__(self, layer_id, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask)
        hidden_states += residual                                            # 残差①
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 残差②
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 数据流第一站
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)        # 尾部唯一"出口归一"
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings,
                                                    rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 推理续写偏移
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0, 0] == 0:   # meta-device 初始化后重算 RoPE 表
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim,
                                                        end=self.config.max_position_embeddings,
                                                        rope_base=self.config.rope_theta,
                                                        rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],
                               self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, position_embeddings,
                                           past_key_value=past_key_value, use_cache=use_cache,
                                           attention_mask=attention_mask)
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = hidden_states.new_zeros(1).squeeze()   # 稠密模型恒为 0，兼容槽位
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config=None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 出口评分器
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight   # L1 §5：同一张 tensor
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])   # [.. , vocab]
        loss = None
        if labels is not None:                                       # 训练：预测下一 token 交叉熵
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                         past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """自回归生成（每步只喂新 token，历史全在 KV cache —— 我们亲口拆过的循环）。"""
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        return input_ids


if __name__ == "__main__":
    # 自检：实例化 → 参数审计 → 前向 loss → 生成路径（全部 CPU，十几秒）
    torch.manual_seed(0)
    cfg = MiniMindConfig()                      # 默认即 zzmind-0.5B 定稿
    model = MiniMindForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n == 497812480, f"参数量不对: {n}"
    print(f"[smoke] ✓ params = {n:,} ≈ 497.8M（E2 审计一致）")

    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(input_ids=ids, labels=ids)
    loss = out.loss.item()
    assert abs(loss - math.log(cfg.vocab_size)) < 0.6, f"初始 loss 离 ln(vocab) 太远: {loss}"
    print(f"[smoke] ✓ loss = {loss:.3f}（期望 ≈ ln(6400)=8.76，随机初始化 ±0.3~0.5 正常）")

    gen = model.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=4, do_sample=False,
                         temperature=1.0, top_k=0, top_p=1.0)
    print(f"[smoke] ✓ generate（KV cache 路径）OK，输出 shape = {tuple(gen.shape)}")
    print("[smoke] 模型自检全部通过 —— zzmind-0.5B 可进入训练阶段。")