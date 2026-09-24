# E3 手写迷你 GPT（对应 02 README 实验任务 E3）
#
# 目的：不看 model_minimind.py，自己写一个 2 层、64 维、小词表的 GPT，
#       在 CPU 上训练"固定模式" a→b→a→b，loss 降下来且预测正确即通过。
# 运行：python E3_mini_gpt.py              （任意机器，纯 CPU，~30 秒）
# 自包含：不依赖 minimind / transformers，只需 torch
# 通过标准：
#   1. 训练 loss 从 ~ln(2)≈0.69 明显下降
#   2. 给 "a b a b"，预测出 "b a b a"
#   3. 能讲清：causal mask、残差、norm 位置 各自在代码哪一行、起什么作用（写进 notes.md）

import sys
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VOCAB, D, HEADS, LAYERS, SEQ, STEPS = 2, 64, 4, 2, 16, 400
A, B = 0, 1


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2)+eps) * w：只缩放不归零"""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w


class Block(nn.Module):
    """x = x + Attn(RMSNorm(x));  x = x + FFN(RMSNorm(x))   ← 残差在两个 + 上"""
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h, self.d = h, d

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q = q.view(b, s, self.h, d // self.h).transpose(1, 2)
        k = k.view(b, s, self.h, d // self.h).transpose(1, 2)
        v = v.view(b, s, self.h, d // self.h).transpose(1, 2)
        mask = torch.full((s, s), float("-inf")).triu(1)          # ← causal：只看左边
        attn = torch.softmax(q @ k.transpose(-2, -1) / (d // self.h) ** 0.5 + mask, dim=-1)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(b, s, d))
        return x + self.ff(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab, d, h, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight                       # ← tie：省一份参数

    def forward(self, ids):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def make_data(n=1024):
    """随机起点 + a b a b 交替，每条 SEQ+1 个 token（前 SEQ 个是输入，后 1 个是目标）"""
    rows = []
    for s in torch.randint(0, VOCAB, (n,)).tolist():
        row = [s]
        for _ in range(SEQ):
            row.append(row[-1] ^ 1)
        rows.append(row)
    return torch.tensor(rows)


@torch.no_grad()
def test(model, label=""):
    ids = torch.tensor([[A, B, A, B]])
    out = model(ids)[:, :-1, :]          # 位置 0~2 分别预测位置 1~3
    pred = out.argmax(-1).squeeze(0).tolist()
    expect = [B, A, B]
    ok = pred == expect
    print(f"[E3] 输入 a b a b  预测 {' '.join('ab'[p] for p in pred)}  "
          f"期望 {' '.join('ab'[p] for p in expect)}  ->  {'PASS' if ok else 'FAIL'} {label}")
    return ok


def main():
    torch.manual_seed(0)
    model = MiniGPT(VOCAB, D, HEADS, LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[E3] MiniGPT: layers={LAYERS} d={D} heads={HEADS} vocab={VOCAB} params={n_params:,}")

    data = make_data()
    x, y = data[:, :-1], data[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, STEPS + 1):
        perm = torch.randperm(len(x))[:128]
        loss = loss_fn(model(x[perm]).reshape(-1, VOCAB), y[perm].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"[E3] step {step:4d}  loss = {loss.item():.4f}   (初始参考 ln2 = 0.6931)")

    final = loss_fn(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    print(f"[E3] 最终 loss = {final:.4f}")
    assert final < 0.2, "loss 没降下来：检查 causal mask / 残差 / 标签是否错位"
    assert test(model)
    print("[E3] PASS：把训练曲线 3~5 个点 + 预测结果贴回 notes.md，并回答'causal/残差/norm 各在哪一行'")


if __name__ == "__main__":
    main()
