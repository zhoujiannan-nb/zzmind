# E2 参数审计（对应 02 README 实验任务 E2 / 可视化图③）
#
# 目的：统计 zzmind-0.5B 每个模块的参数量与占比，与手算值对账。
# 运行：python E2_param_audit.py            （node05 或本地 Windows 都能跑，CPU 即可）
# 环境变量：MINIMIND_DIR 指向 minimind 仓库根目录（默认自动探测）
# 通过标准：
#   1. 每层 ≈ 20.4M（FFN 15.48M + 注意力 4.92M + norm ≈ 0.003M）
#   2. 总计 ≈ 498M，与手算偏差 < 5%
#   3. 把表格贴回 notes.md / DESIGN.md 的"参数核对"一节

import os, sys, json
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
    raise SystemExit("[E2] 找不到 minimind 代码，请设置环境变量 MINIMIND_DIR 指向 minimind 根目录")


def M(n):
    return f"{n / 1e6:8.3f}M"


def main():
    minimind_dir = find_minimind()
    sys.path.insert(0, minimind_dir)
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "design", "config_zzmind0.5b.json"))
    raw = json.load(open(cfg_path, encoding="utf-8"))
    cfg = MiniMindConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})
    torch.manual_seed(0)
    model = MiniMindForCausalLM(cfg)

    def numel(mod):
        return sum(p.numel() for p in mod.parameters())

    l0 = model.model.layers[0]
    att, ffn = l0.self_attn, l0.mlp
    rows = [
        ("gate_proj  ", numel(ffn.gate_proj)),
        ("up_proj    ", numel(ffn.up_proj)),
        ("down_proj  ", numel(ffn.down_proj)),
        ("q_proj     ", numel(att.q_proj)),
        ("o_proj     ", numel(att.o_proj)),
        ("k_proj     ", numel(att.k_proj)),
        ("v_proj     ", numel(att.v_proj)),
        ("q_norm     ", numel(att.q_norm)),
        ("k_norm     ", numel(att.k_norm)),
        ("in_ln      ", numel(l0.input_layernorm)),
        ("post_ln    ", numel(l0.post_attention_layernorm)),
    ]
    per_layer = sum(n for _, n in rows)
    ffn_total = rows[0][1] + rows[1][1] + rows[2][1]
    att_total = sum(n for _, n in rows[3:9])
    nrm_total = per_layer - ffn_total - att_total

    total = numel(model)          # tied 权重只计一次（lm_head 与 embedding 同一块）
    emb = numel(model.model.embed_tokens)

    print(f"[E2] zzmind-0.5B 参数审计  (L={cfg.num_hidden_layers}, h={cfg.hidden_size}, "
          f"kv={cfg.num_attention_heads}:{cfg.num_key_value_heads}, ffn={cfg.intermediate_size}, "
          f"vocab={cfg.vocab_size}, tie={cfg.tie_word_embeddings})")
    print("-" * 60)
    print("单层拆解（× %d 层）:" % cfg.num_hidden_layers)
    for name, n in rows:
        print(f"  {name} {M(n)}  {n / per_layer * 100:5.2f}%")
    print(f"  每层合计   {M(per_layer)}")
    print("-" * 60)
    print(f"  FFN 小计        {M(ffn_total)}   占每层 {ffn_total / per_layer * 100:5.2f}%")
    print(f"  注意力小计      {M(att_total)}   占每层 {att_total / per_layer * 100:5.2f}%")
    print(f"  norm 小计       {M(nrm_total)}   占每层 {nrm_total / per_layer * 100:5.2f}%")
    print("-" * 60)
    print(f"  {cfg.num_hidden_layers} 层合计     {M(per_layer * cfg.num_hidden_layers)}")
    print(f"  Embedding      {M(emb)}  (lm_head tied，不重复计)")
    print(f"  final norm     {M(numel(model.model.norm))}")
    print(f"  >>> 总计       {M(total)}  = {total:,} 个参数")
    print("-" * 60)

    # 与手算对账
    hand_per_layer = (3 * cfg.hidden_size * cfg.intermediate_size
                      + cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim
                      + 2 * cfg.hidden_size * cfg.num_key_value_heads * cfg.head_dim
                      + cfg.hidden_size * cfg.head_dim * cfg.num_attention_heads)  # q/o/k/v
    hand_total = hand_per_layer * cfg.num_hidden_layers + cfg.vocab_size * cfg.hidden_size
    dev = abs(total - hand_total) / hand_total * 100
    print(f"  手算每层 ≈ {hand_per_layer / 1e6:.3f}M (README 给 20.4M)")
    print(f"  手算总计 ≈ {hand_total / 1e6:.1f}M (README 给 ~498M)")
    print(f"  实测 vs 手算 偏差 = {dev:.2f}%  ({'PASS < 5%' if dev < 5 else 'FAIL >= 5%'})")
    print("[E2] 把上面表格贴回 notes.md，并把 手算/实测/偏差 填进 design/DESIGN.md")


if __name__ == "__main__":
    main()
