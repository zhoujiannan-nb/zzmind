# DESIGN：zzmind-0.5B 定稿架构 v1.0

> 状态：**拍板定稿**（2026-09-28）｜依据：L0 约束 → L1 骨架 → L2 容量，全部推导文档在 design/ 下
> 与旧工作坊"均衡方案 B"完全一致 → 独立推导收敛到同一解，交叉验证了设计正确性。

## 1. 任务定位（架构为什么长这样）

minicode = 0.5B 小智能体：**人格陪伴**（风格稳定）+ **2~4 个工具调用**（格式输出）+
**轻量推理**（think token 短思维链）+ **多意图路由**（data/ 语料）。
→ 架构取舍：GQA 控推理 KV cache、深度足（多步链）、FFN 厚（格式/知识记忆）、
RoPE 长上下文（以后 agent 多轮要用）。

## 2. 定稿规格（`config_zzmind0.5b.json`，已冻结）

| 超参 | 值 | 一句话依据（详见） |
|---|---|---|
| hidden_size | 1280 | L2：均衡解宽度，attn:ffn≈1:3.15 |
| num_hidden_layers | 24 | L2：多步链深度优先 + 0.5B 档已验证密度 |
| num_attention_heads / KV | 16 / 8 | L1§1 GQA：KV cache 省一半 |
| head_dim | 80 | 1280÷16 |
| intermediate_size | 4032 | L2：ceil(H·π/64)·64，SwiGLU 配比系数 |
| vocab_size | 6400 | L0 C3：01 的 BPE 锁死 |
| max_position_embeddings | 32768 | L0 C4：长上下文上限 |
| rope_theta | 1e6 | L1§2：对齐 minimind 源码 / Qwen3 |
| rms_norm_eps | 1e-6 | minimind 默认 |
| hidden_act | silu | L1§4：SwiGLU 门控 |
| tie_word_embeddings | true | L1§5：省 8.19M + 同空间正则 |
| dropout / bias | 0.0 / False | L1§6：欠训练阶段不配正则，现代标配去 bias |
| flash_attn | true | torch SDPA，预训练提速 |
| use_moe | false | 0.5B 全参足够；字段为 minimind 兼容保留 |

## 3. 结构（一块 Block = 10 个组件）

```
RMSNorm → Attention(GQA 16Q/8KV, QK-Norm, RoPE) → +residual
RMSNorm → MLP(SwiGLU: silu(gate)·up → down) → +residual
```
整体：Embedding(6400×1280, 与 lm_head 共享) → 24×Block → final RMSNorm → lm_head → logits

## 4. 参数量（E2 验证通过）

| 组件 | 参数 |
|---|---|
| embedding（共享，只算一次） | 8,192,000（1.6%） |
| 每块：attention 4,915,360 + mlp 15,482,880 + 2×norm 2,560 | 20,400,800 |
| ×24 层 | 489,619,200 |
| final RMSNorm | 1,280 |
| **总计** | **497,812,480 ≈ 497.8M**（偏差 0.44%） |

## 5. 关键账

- 训练：6·N·T ≈ 4.5e18 FLOPs，2×4090 bf16 → **15~28h**（1.5B token 首版）
- 训练显存：~8G/卡系 + 激活，24G×2 宽裕（bf16 全参 DDP）
- 推理：32k 上下文 KV cache 1.88G/条（GQA-8，MHA 将是 3.75G）
- 已知欠训练（见 L2§6）：先跑通，必要时补训全量语料，**不动架构**

## 6. 验证状态（L3）

- [x] E1 forward walk：本地 CPU 跑通，shape 正确；初始 loss **9.06** ≈ ln(6400)=8.76
  （差 0.30 = 随机初始化 logits 非全零的正常波动，验收线定为"与 ln(vocab) 差 <0.5"）
- [x] E2 param audit：**497,812,480 = 497.81M**，手工账 vs 实例化偏差 0.000%，
  相对 500M 预算 99.56%（C1 通过）
- [x] E3 mini GPT：110,912 参数的最小 decoder 纯 torch 从 0 学"后继规则"，loss 4.72 → 0.026
- [ ] node05 训练前：用同样本 E2 在 GPU 环境复核一次

## 7. 变更记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-09-28 | 定稿；与旧工作坊"方案 B"逐项一致（GQA 16:8 / 1280×24 / 4032 / RoPE 1e6 / tie） |