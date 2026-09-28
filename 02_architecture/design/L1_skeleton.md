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
## 3. 归一化 → ✅ RMSNorm（pre-norm），"为什么 pre" 待讲

### 3.1 RMSNorm 地基
**名字**：RMS = Root Mean Square（均方根），Norm = 归一化。
**一句话**：把每个 token 的 1280 维向量除以它自己的均方根（音量拉回标准刻度），再乘可学习的逐维缩放。

minimind 代码逐行拆解：
```python
def norm(self, x):        # x: [B, S, 1280]
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```
| 步骤 | 含义 |
|---|---|
| `x.pow(2).mean(-1)` | 1280 个数平方取平均 = "均方"（≈向量能量） |
| `rsqrt(· + eps)` | 开方取倒数 = 除以 RMS；eps 防除零 |
| `x * ...` | 操作后向量 RMS ≈ 1（音量统一） |
| `self.weight *` | 1280 个可学习缩放（初始 1）= "均衡器" |
| `.float()`/`.type_as` | fp32 计算防精度丢失，转回 bf16 |

**LayerNorm vs RMSNorm**：LN 干两步（①减均值 ②除标准差）+bias；
RMSNorm 只干②，**少的就是"减均值"**，bias 也没了。
论文（Zhang & Sennrich 2019）实测不减均值质量基本无损 → 留快的。

**为什么必须有**：每层是大矩阵乘，输出"音量"随初始化/数据漂移；
24 层堆叠后有的层输入炸响（梯度爆）、有的轻到消失（梯度没）。
RMSNorm = 每个子层入口的"音量旋钮"，weight = 均衡器。

**参数**：只有 weight 无 bias（所以参数表里 norm 只算 1280/80）。
全模型 97 个（块内 24×2 + final 1 + q/k_norm 24×2）≈ 66.5K 参数。

### 3.2 为什么 pre-norm（答）
```
pre:  x = x + Attention(RMSNorm(x))     ← norm 在残差"进"子层之前
post: x = RMSNorm(x + Attention(x))     ← norm 压在残差"合"回来之后
```
盯残差直通高速公路：
- **pre-norm**：`x = x + Δ` 这条恒等路径是**纯的**——梯度回流走 `∂x/∂x = 1`，
  中间没有 norm、没有缩放。每个子层只读一份**归一化拷贝** RMSNorm(x)、写回一个小增量 Δ。
  24 层里 x 像"内存总线"一样干净地累积信息（业内叫 residual stream = 未归一化的记忆）。
- **post-norm**：norm 压在**整个和**上，残差路的梯度必须穿过 RMSNorm 的导数（非恒等），
  而且每层都把累积的记忆**裁剪/重标定**一次 → 穿 24 次 norm，梯度一路打折 → 难训。
- 实证：原始 Transformer 是 post-norm，必须重 warmup + 小 lr 才训得动；LLaMA 起全线 pre-norm，
  可以直接上高 lr、少 warmup。模型顶部的 final RMSNorm 是唯一例外的"尾部归一"，只在出最后一层时做一次。
**一句话**：pre-norm 的残差路保持"干净恒等"，堆多少层梯度都不打折；post-norm 把梯度路和记忆都污染了。
## 4. FFN 激活 → ✅ SwiGLU（silu 门控，3 张矩阵）

```
GELU 旧路线：down( gelu( up(x) ) )          2 张矩阵，每条中间神经元"全开"
SwiGLU 现役：down( silu( gate(x) ) ⊙ up(x) ) 3 张矩阵，gate 决定每条神经元"开多大"
```
- **门控是核心**：attention 有软门（softmax），FFN 也配一把闸——`silu(gate(x))` 每个元素 ∈(0,x)，
  控制 up(x) 里每条"知识线路"放行多少。论文（Shazeer 2020 GLU 变体）实测在同等预算下
  门控路线比 ReLU/GELU 质量更好，现在 LLaMA/Qwen 全系标配。
- silu(x)=x·σ(x)：平滑、无死神经元、梯度好。
- **代价**：多一张矩阵（gate）。我们为它买单——见 L2，FFN 拿总预算是大头，值得。
- 参数：3×H×I = 15.48M/块（占 Block 的 76%），这是"知识库"本体；
  attention 是"混音器"只占 ~24%，配比 ≈1:3.15。
- I=4032 的来历（minimind 源码 line26 就是这个公式）：`I = ceil(H·π/64)·64 = ceil(1280×3.1416/64)×64 = 4032`
  ——π 是配比经验系数（≈3.14），64 对齐 tensor-core 最小粒度。

## 5. embedding 与 lm_head 共享 → ✅ tie_word_embeddings = true

- 不是"复制"，是**同一张 tensor**：`model.embed_tokens.weight = lm_head.weight`。
- 省 V×H = 6400×1280 = **8.19M**（全模型第二大项，仅次于 24×FFN）。
- 额外收益：输出 logits 与输入同空间 = 天然正则（词典不会漂出输入几何）；
  词表小（6400）+ 同一中文分布下几乎没有损失。LLaMA-2/Qwen 小模型都绑。

## 6. 训练技巧 → ✅ dropout=0.0、bias=False、flash_attn=True

- **dropout 0.0**：0.5B 在 4~5B token 语料上是**欠训练**（研究目标明确：先会说话+格式），
  dropout 只拖慢收敛；正则需求交给 04/05 后训练阶段去配。
- **bias=False 全关**：现代标配，省参数、不与归一化抢尺度。
- **flash_attn=True**：用 torch SDPA（FlashAttention 后端），预训练提速；代码 hasattr 防旧版本。

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
