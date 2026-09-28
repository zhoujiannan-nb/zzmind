#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 · param audit：按 DESIGN.md 逐组件核对参数量，并真实实例化模型对账。
期望：总计 497,812,480 ~= 497.8M，偏差 < 2%（验收线）。
注意：tie 后 lm_head 与 embed_tokens 是同一张 tensor，必须用 named_parameters() 数（去重），
      用 state_dict 会把共享权重数两遍（+8.19M 假账）。
用法：python E2_param_audit.py
依赖：minimind 底座（默认 D:\\project_ai\\minimind，可用环境变量 MINIMIND_ROOT 覆盖）
"""
import os, sys, json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MM = os.environ.get("MINIMIND_ROOT", r"D:\project_ai\minimind")
sys.path.insert(0, MM)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = os.path.join(HERE, "..", "design", "config_zzmind0.5b.json")
TARGET = 500_000_000  # C1 预算（容差 450M~550M，验收线=与拍板值偏差<2%）

def load_config():
    raw = json.load(open(CFG, encoding="utf-8"))
    skip = {"_model", "_version", "_task", "_design_ref", "model_type"}
    return MiniMindConfig(**{k: v for k, v in raw.items() if k not in skip})

def main():
    cfg = load_config()
    H, L, I = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    V, hd = cfg.vocab_size, cfg.head_dim
    kv = cfg.num_key_value_heads

    # ---- 手工账（DESIGN.md §4 同源公式）----
    attn = 2*H*H + 2*H*kv*hd + 2*hd                # q/o + k/v + qk_norm
    ffn  = 3*H*I                                   # gate/up/down
    block = attn + ffn + 2*H                       # + input/post_attention_layernorm
    manual = V*H + L*block + H                     # embedding + 24×block + final norm
    print(f"手工账：attn/层 {attn:,}  ffn/层 {ffn:,}  块 {block:,}  总计 {manual:,} = {manual/1e6:.2f}M")

    # ---- 真实实例化对账（named_parameters 去重 tied 权重）----
    model = MiniMindForCausalLM(cfg)
    per = {}
    for name, p in model.named_parameters():
        group = "embed" if ("embed_tokens" in name or name.startswith("lm_head")) else \
                "attn"  if ".self_attn." in name else \
                "ffn"   if ".mlp." in name else \
                "norm"  if ".layernorm." in name or "model.norm" in name else "other"
        per[group] = per.get(group, 0) + p.numel()

    total = sum(per.values())
    dev = (total - manual) / manual * 100
    print("实例化对账：")
    for g in ("embed", "attn", "ffn", "norm"):
        print(f"  {g:>6}: {per.get(g,0):>12,}  ({per.get(g,0)/total*100:.1f}%)")
    print(f"  总计: {total:,} = {total/1e6:.2f}M")
    print(f"  与手工账偏差: {dev:+.3f}%")
    print(f"  相对 500M 预算: {total/TARGET*100:.2f}% (C1 容差 450~550M {'OK' if 450e6<=total<=550e6 else 'FAIL'})")
    assert abs(dev) < 0.1, "实例化与手工账对不上!"
    assert abs(total - 497812480) < 10, "与 DESIGN.md 拍板值不一致!"
    print("[OK] E2 PASSED (497.8M, deviation <1%)")

if __name__ == "__main__":
    main()