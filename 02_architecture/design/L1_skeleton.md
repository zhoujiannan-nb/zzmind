# L1 · 骨架选择（六个组件逐个拍板）

> 方法：每个组件写 **选择 + 理由 + 落选方案 + 数字影响**。
> 选择必须不违反 L0 的 C1~C4。

## 1. 注意力类型：MHA / GQA / MQA → ✅ GQA（16 Q / 8 K + 8 V）已确认

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

## 2. 位置编码 → ✅ RoPE（base 1e6，max_pos 32768）

### 为什么需要：attention 对词序"失明"
纯 attention 打分只看 Q/K 内容，是**置换不变**的：把输入打乱顺序，
输出只是同样打乱——"猫 打 了 狗" 和 "狗 了 打 猫" 的 token 表示完全相同。
词序 = 语法 = 意义，必须从外部注入位置信息。

### 三条路线
| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 绝对位置 embedding（原始 Transformer） | 每个位置学一个向量，加到 token embedding 上 | 简单 | 训练长度是上限；编的是"绝对位置"，模型要自己学"距离" |
| ALiBi（LLaMA1） | attention 分数上加 -m·\|i-j\| 惩罚 | 天然相对、易外推 | 惩罚强度固定不可学 |
| **RoPE（LLaMA2/3、Qwen3）** | **Q/K 按位置旋转** | **天然相对、零参数、可外推（YaRN）** | 数学稍绕 |

### RoPE 核心直觉（唯一的 "aha" 点）
每个 Q/K 头是 80 维（head_dim）。RoPE 把它拆成 **40 对二维平面**，
第 i 对平面按角度 `pos × θᵢ` 旋转，θᵢ = base^(-2i/80)，base = 1e6
（40 个"频率"从高到低：周期 6.28 → ~440 万，像 40 根转速不同的钟针）。

**为什么是"相对"？** 靠旋转矩阵的正交性：
```
⟨R_m·q, R_n·k⟩ = qᵀ R_mᵀ R_n k = ⟨q, R_{n-m} k⟩
```
q 在位置 m 旋转、k 在位置 n 旋转 → 点积里绝对角全部抵消，
**只剩差 n-m**。attention 只看到"谁离谁多远、谁前谁后"。

钟类比：看两个时刻的夹角，不需要知道"现在几点"（绝对），
只需要看**指针夹角**（相对距离）；多档转速组合能唯一区分任意距离。

### 三个推论
1. **零参数**：旋转角是固定公式（θ 表不用学）→ L0 C4 的"白拿"就是它。
2. **长上下文**：base 1e6 时最慢频率周期 ~440 万，32k 覆盖余量充足；
   将来再扩长就 YaRN/NTK 缩 θ（minimind `precompute_freqs_cis` 已内置 YaRN），不用从头重训。
3. **Q 和 K 必须同时转**：都转，绝对角才抵消；只转一个，绝对位置就漏进来。
   **V 不转**：V 只参与"内容取回"（加权和），不参与打分。

### 决定
RoPE，base = **1e6**（对齐 minimind 源码 `precompute_freqs_cis(rope_base=1e6)` / Qwen3；
不用 1e4：1e4 最慢频率周期仅 ~5 万，对 32k 偏紧），max_pos = 32768。
## 3. 归一化 ⬜（RMSNorm pre-norm vs LayerNorm）
## 4. FFN 激活 ⬜（SwiGLU vs GELU）
## 5. embedding 与 lm_head 是否共享（tie）⬜
## 6. 训练技巧（dropout 等，0.5B 下基本全关）⬜

## 7. 整体解剖（组件全景）

> 回答"从上到下第一层是不是 attention"：**分两层说**。
> 数据流第一站是 **Embedding**（不是 attention）；所谓"层"（num_hidden_layers=24）
> 是一个 **Block**，Block 内部第一子层才是 Attention。

```
input_ids [B,S]（token 编号，整数）
  │
  ▼
① embed_tokens：查表 (6400×1280) → [B,S,1280]          ← 数据流第一站
  │
  ▼
② 24 × Block：
  │  x = x + Attention(RMSNorm(x))                       ← attn 子层（在前）
  │       └ q/k/v_proj → q_norm/k_norm(QK-Norm) → RoPE → SDPA → o_proj
  │  x = x + MLP(RMSNorm(x))                             ← MLP 子层（在后）
  │       └ silu(gate(x))·up(x) → down(x)  （SwiGLU）
  │
  ▼
③ final RMSNorm
  │
  ▼
④ lm_head（1280→6400，与 ① 共享权重）→ logits [B,S,6400]
```

minimind 源码坐实（`model_minimind.py`）：
- **pre-norm**：norm 在每个子层前面，外层 residual 包住（MiniMindBlock.forward）
- **QK-Norm**：q_norm/k_norm = RMSNorm(head_dim=80)，作用在 Q/K 上、**RoPE 之前**
  ——主 README 里写的 "QK-Norm" 就是这两行
- **tie**：`embed_tokens.weight = lm_head.weight`，同一张 tensor

参数初算（L2 的第一次粗算，"为什么是这些值"留 L2 讨论）：

| 组件 | 参数 |
|---|---|
| Attn：Q 1.64M + K 0.82M + V 0.82M + O 1.64M + q/k_norm 160 | 4.92M |
| MLP：gate/up/down × 5.16M | 15.48M |
| 2×RMSNorm | 2,560 |
| **一个 Block** | **≈ 20.4M** |
| × 24 层 | 489.6M |
| Embedding（与 lm_head 共享，只算一次）6400×1280 | 8.19M |
| final RMSNorm | 1,280 |
| **总计** | **≈ 497.8M ✅（落在 C1 预算 450~550M）** |

主 README "≈498M" 的出处就是这张表。
