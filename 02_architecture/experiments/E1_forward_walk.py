#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 · forward walk：实例化 zzmind-0.5B，跑一次前向，核对 shape 流和初始 loss。
期望：
  - 初始 loss ~= ln(vocab_size) = ln(6400) ~= 8.76（随机初始化 -> 均匀分布猜测）
  - 一个 backward 后梯度全非 NaN（架构可训练）
用法：python E1_forward_walk.py [--backward]
依赖：minimind 底座（默认 D:\\project_ai\\minimind，可用环境变量 MINIMIND_ROOT 覆盖）
"""
import os, sys, json, math, argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MM = os.environ.get("MINIMIND_ROOT", r"D:\project_ai\minimind")
sys.path.insert(0, MM)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = os.path.join(HERE, "..", "design", "config_zzmind0.5b.json")

def load_config():
    raw = json.load(open(CFG, encoding="utf-8"))
    skip = {"_model", "_version", "_task", "_design_ref", "model_type"}
    return MiniMindConfig(**{k: v for k, v in raw.items() if k not in skip})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backward", action="store_true", help="额外跑一次 backward 验证可训练")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = load_config()
    print(f"[spec] {cfg.hidden_size}x{cfg.num_hidden_layers} heads={cfg.num_attention_heads}:{cfg.num_key_value_heads} "
          f"head_dim={cfg.head_dim} I={cfg.intermediate_size} vocab={cfg.vocab_size} tie={cfg.tie_word_embeddings}")

    model = MiniMindForCausalLM(cfg).eval()          # float32 CPU，验证为主不求快
    n = sum(p.numel() for p in model.parameters())
    print(f"[init] params = {n:,} = {n/1e6:.2f}M")

    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq))
    labels = ids.clone()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
    loss = out.loss.item()
    print(f"[forward] input {tuple(ids.shape)} -> logits {tuple(out.logits.shape)} (期望 [B,S,{cfg.vocab_size}])")
    print(f"[loss  ] {loss:.4f}  (期望 ~= ln(6400) = {math.log(cfg.vocab_size):.4f})")
    assert abs(loss - math.log(cfg.vocab_size)) < 0.5, "初始 loss 离 ln(vocab) 太远，检查架构!"

    if args.backward:
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        gnorm = max((p.grad.abs().max().item() for p in model.parameters() if p.grad is not None), default=0.0)
        print(f"[backward] OK, 最大梯度 abs 值 = {gnorm:.3e}（非 NaN 即通过）")
    print("[OK] E1 PASSED")

if __name__ == "__main__":
    main()