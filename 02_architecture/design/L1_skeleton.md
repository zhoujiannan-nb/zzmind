# L1 · 骨架选择（六个组件逐个拍板）

> 方法：每个组件写 **选择 + 理由 + 落选方案 + 数字影响**。
> 选择必须不违反 L0 的 C1~C4。

## 1. 注意力类型：MHA / GQA / MQA → ✅ 选 GQA（16 Q / 8 KV），待确认

### 三者的唯一区别
16 个 Q 头固定（"阅读器"数量），区别只在于配多少套 K/V 头（"词典"）：

| 变体 | Q 头 | KV 头 | 共享关系 |
|---|---|---|---|
| MHA | 16 | 16 | 每个 Q 头独享一套 K/V |
| GQA | 16 | 8 | 2 个 Q 头共享一套 K/V（8 组） |
| MQA | 16 | 1 | 16 个全共享一套 K/V |

**为什么只动 KV 不动 Q**：Q 是"一次性"的——decode 每步只算当前 token 的 Q，
历史 Q 不存；K/V 是"累积"的——新 token 要 attend 全部历史，每个历史 token 的
K、V 必须存（KV cache，随 seq 线性增长）。成本全在 K/V 侧，
砍 Q 头省不了缓存，只会丢"读信息的视角数"。

### KV cache（bf16，32k 上下文，24 层，head_dim 80）
`= 2(K,V) × 24 层 × KV 头数 × 80 × 2B × seq`

| | 每 token | 32k 单序列 | ×8 并发 |
|---|---|---|---|
| MHA (16) | 120 KB | 3.75 G | 30 G ← 24G 卡装不下 |
| GQA (8) | 60 KB | 1.88 G | 15 G ← 能装 |
| MQA (1) | 7.5 KB | 234 M | 1.9 G |

decode 是带宽瓶颈（每步把整个 KV cache 读一遍）→ cache 砍半 = 生成直接提速。
实现上 GQA 的 8 套 KV 会 expand 成 16 份做 matmul（或分组 bmm），但**存的只有 8 套**。

### 训练侧（为什么"几乎没感觉"）
- attention 计算量由 16 个 Q 头决定，三个变体相同
- K/V 投影权重：MHA 78.6M / GQA 39.3M / MQA 4.9M（24 层合计）
  —— GQA 只省 39M（~8%），L2 里把这笔预算投回其他维度
- 训练显存大头 = 激活 + 优化器状态，与 KV cache 无关

### 质量与决定
- 容量：MHA ≥ GQA > MQA（KV 头 = 记忆编码维度，越少越省但表达能力越弱）
- GQA 论文（Ainslie et al. 2023）：g=8 基本追回 MHA 质量，MQA 有小掉点
- 业界：LLaMA2-70B / LLaMA3 全系 GQA
- **决定：GQA，KV 头 = Q/2 = 8**。32k KV cache 3.75G→1.88G，质量代价≈0，
  与主 README "方案 B" 表一致。

## 2. 位置编码 ⬜（RoPE vs 绝对位置 vs ALiBi）
## 3. 归一化 ⬜（RMSNorm pre-norm vs LayerNorm）
## 4. FFN 激活 ⬜（SwiGLU vs GELU）
## 5. embedding 与 lm_head 是否共享（tie）⬜
## 6. 训练技巧（dropout 等，0.5B 下基本全关）⬜
