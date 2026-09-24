# E1 前向走查（对应 02 README 实验任务 E1 / 可视化图②）
#
# 目的：随机初始化的 zzmind-0.5B 喂一句话，打印每个位置的张量形状、
#       logits 范围、初始 loss，验证你对"形状流"的理解。
# 运行：python E1_forward_walk.py            （node05 有 GPU 自动用 GPU；本地 Windows CPU 也能跑）
# 环境变量：MINIMIND_DIR 指向 minimind 仓库根目录（默认自动探测）
# 通过标准：
#   1. 每个位置形状 = fig2_flow_shapes.png 上的标注（[1, S, 1280] / 头切分 [1, S, 16, 80] 等）
#   2. 初始 loss ≈ ln(6400) ≈ 8.76（随机 init ≈ 均匀分布）
#   3. 手算形状与打印值逐一对照（把输出贴回 notes.md）

import os, sys, json, math
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MINIMIND_CANDIDATES = [
    os.environ.get("MINIMIND_DIR", ""),
    r"D:\project_ai\minimind",
    os.path.expanduser("~/minimind"),
    "/root/minimind",
    "/mnt/boot/minimind",
]


def find_minimind():
    for p in MINIMIND_CANDIDATES:
        if p and os.path.isfile(os.path.join(p, "model", "model_minimind.py")):
            return p
    raise SystemExit("[E1] 找不到 minimind 代码，请设置环境变量 MINIMIND_DIR 指向 minimind 根目录")


def load_config(path):
    raw = json.load(open(path, encoding="utf-8"))
    from model.model_minimind import MiniMindConfig
    return MiniMindConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})


def main():
    minimind_dir = find_minimind()
    sys.path.insert(0, minimind_dir)
    from model.model_minimind import MiniMindForCausalLM
    from transformers import AutoTokenizer

    cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "design", "config_zzmind0.5b.json"))
    cfg = load_config(cfg_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(42)
    model = MiniMindForCausalLM(cfg).to(device=device, dtype=dtype).eval()

    tok = AutoTokenizer.from_pretrained(os.path.join(minimind_dir, "model"))
    sentence = "我们是一个两人小组，从 0 开始训练一个 0.5B 的小智能体。"
    ids = tok(sentence, return_tensors="pt")["input_ids"].to(device)
    S = ids.shape[1]
    print(f"[E1] 句子: {sentence}")
    print(f"[E1] 词表={cfg.vocab_size} 层数={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} head_dim={cfg.head_dim} ffn={cfg.intermediate_size}")
    print(f"[E1] S={S}  device={device}  dtype={dtype}")
    print("-" * 70)

    with torch.no_grad():
        m = model.model
        h = m.dropout(m.embed_tokens(ids))
        print(f"  input_ids      {tuple(ids.shape)}  (整数 0~{cfg.vocab_size-1})")
        print(f"  Embedding      -> h {tuple(h.shape)}   (6400x1280 查表, 与 lm_head 共享)")

        cos, sin = m.freqs_cos[:S], m.freqs_sin[:S]
        # 逐层走查：每层 = x + attn(x) 然后 x + ffn(x)，输出应保持 [1, S, 1280]
        for i, layer in enumerate(m.layers):
            h, _ = layer(h, (cos, sin))
            print(f"  layer{i:2d} out    {tuple(h.shape)}   mean={h.float().mean().item():+.4f} std={h.float().std().item():.4f}")

        h = m.norm(h)
        print(f"  final RMSNorm  -> h {tuple(h.shape)}")
        logits = model.lm_head(h)
        lf = logits.float()
        print(f"  lm_head        -> logits {tuple(logits.shape)}   "
              f"min={lf.min().item():.2f} max={lf.max().item():.2f} |mean|={lf.abs().mean().item():.4f}")

        out = model(ids, labels=ids)
        loss = out.loss.item()
    print("-" * 70)
    print(f"  初始 loss = {loss:.4f}    均匀分布参考 ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")
    assert abs(loss - math.log(cfg.vocab_size)) < 0.5, "初始 loss 偏离均匀分布太多，检查初始化/词表大小"
    print("[E1] PASS：把上面的形状表逐行与 fig2 对照后，贴回 notes.md")


if __name__ == "__main__":
    main()
