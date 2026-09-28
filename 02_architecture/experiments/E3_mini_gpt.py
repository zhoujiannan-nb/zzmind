#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3 · mini GPT：从 0 写一个最小 decoder（只依赖 torch，看懂每个零件在干嘛）。
学习目的：不查 minimind，用最朴素的写法把 DECODER 组装一遍——
  embedding(+简单位置) -> 2×Block(RMSNorm -> Attention -> 残差) -> final norm -> lm_head
任务：学一个"记忆音阶"的最小任务（预测下一个音调），看 loss 真能降。——证明这套零件
      不是抄的、而是确实能学。
用法：python E3_mini_gpt.py
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

torch.manual_seed(0)
V, D, L, H, I = 96, 64, 2, 4, 128          # 词表/隐藏/层数/头数/FFN宽（迷你版，仅演示）

class RMSNorm(nn.Module):                  # 音量旋钮：除以均方根 + 可学习缩放
    def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

class Attn(nn.Module):                     # 混音器：MHA + 因果 mask（位置用简单全局 sin/cos 表示）
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(D,D,bias=False) for _ in range(3))
        self.o = nn.Linear(D,D,bias=False)
    def forward(self, x):
        B,S,_ = x.shape
        q = self.q(x).view(B,S,H,-1).transpose(1,2)
        k = self.k(x).view(B,S,H,-1).transpose(1,2)
        v = self.v(x).view(B,S,H,-1).transpose(1,2)
        a = q @ k.transpose(-2,-1) / (k.shape[-1] ** 0.5)
        mask = torch.triu(torch.ones(S,S,device=x.device,dtype=torch.bool), diagonal=1)
        a = a.masked_fill(mask, float("-inf")).softmax(-1)
        o = (a @ v).transpose(1,2).reshape(B,S,D)   # 先还原成 [B,S,D] 再过 o_proj
        return self.o(o)

class Block(nn.Module):                    # pre-norm：残差走干净恒等，norm 在子层前
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.attn = Attn()
        self.mlp = nn.Sequential(nn.Linear(D,I,bias=False), nn.SiLU(), nn.Linear(I,D,bias=False))
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V,D)
        self.pos = nn.Parameter(torch.randn(512,D) * 0.02)   # 简化绝对位置（看懂即可；真身是 RoPE，见底座）
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.n = RMSNorm(D)
        self.head = nn.Linear(D,V,bias=False)
    def forward(self, ids):
        x = self.emb(ids) + self.pos[:ids.shape[-1]]
        for b in self.blocks: x = b(x)
        return self.head(self.n(x))

model = MiniGPT()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f"miniGPT params = {sum(p.numel() for p in model.parameters()):,}")

def batch(B=8, S=12):                      # 造数据：学"后继规则" y=(x+1)%V —— 有结构、能真学会
    x = torch.randint(1, V, (B,S))
    y = (x + 1) % V
    return x, y

before = None
for step in range(300):
    x, y = batch(); logits = model(x)
    loss = F.cross_entropy(logits.view(-1,V), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if before is None: before = loss.item()
    if step % 60 == 0 or step == 299:
        print(f"step {step:>3}: loss {loss.item():.4f}")
assert loss.item() < before * 0.4, "收敛太慢? 调 lr 或加大步数"
print("[OK] E3 PASSED (最小 decoder 从 0 学下来了)")