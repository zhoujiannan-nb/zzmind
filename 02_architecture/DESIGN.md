# zzmind-0.5B 架构设计 · 定稿 v1.0

> 一个文件讲完"为什么设计成这样"：任务 → 约束 → 组件 → 容量 → 结构 → 参数 → 账单 → 验证 → 配置。
> 讲解版（图文、可折叠）：**`DESIGN.html`**；定稿配置全文见文末 §9。
> 状态：**已定稿并验证**（2026-09-28）｜训练底座：minimind（Qwen3 风格 decoder-only）

---

## 0. 一屏看懂（TL;DR）

| 项 | 值 |
|---|---|
| 规模 | **497.8M 参数**（0.5B 预算的 99.56%，C1 容差 450~550M ✅） |
| 架构 | 24 层 decoder-only Transformer（Qwen3/minimind 风格） |
| 配置 | hidden=1280 ｜ layers=24 ｜ Q/KV 头 = 16/8（head_dim 80）｜ FFN=4032 ｜ 词表=6400 ｜ 上下文=32k |
| 关键取舍 | GQA 省推理内存 ｜ RoPE 白拿长上下文 ｜ RMSNorm pre-norm 深堆稳定 ｜ tie 省 8.2M |
| 验证 | E1 前向 loss≈ln6400 ✅ ｜ E2 参数审计 497.8M（偏差 0.000%）✅ ｜ E3 最小 decoder 从 0 学会 ✅ |

---

## 1. 任务定位（架构为什么长这样）

minicode = 0.5B 小智能体，要会四件事：
1. **人格陪伴**：风格稳定的对话（毒舌/冷静/精准/自信/偶尔俏皮）
2. **2~4 个工具调用**：get_time / calculator / get_weather / search_web…（严格 JSON 格式输出）
3. **轻量推理**：think token 短思维链（QwQ 思路）
4. **多意图路由**：把用户请求路由到合适能力档（data/ 阶段 68 万条语料）

→ 对架构的四个要求：**格式记忆厚**（FFN 拿大头）、**长程交互带宽足**（注意力配比健康）、**多步链深度够**（层数够深）、**推理省内存**（GQA + RoPE 32k）。

### 1.1 为什么是 0.5B（候选排除）

判断锚点 = **迭代成本**（我们要反复换人格/工具重训），不是"越大越好"。

| 规模 | 2×4090 训 1.5B token（粗估） | 换一版人格重训 | 结论 |
|---|---|---|---|
| ~100M | 3~6 h | 一天数版 | 太小：工具格式勉强、性格扁平 |
| **~500M** | **15~28 h** | **一周一两版，可行** | ✅ 甜点 |
| ~7B | 数周起、且严重欠训练 | 按月上计 | 迭代太慢，杀"性格实验自由" |

---

## 2. 硬约束（设计宪法 C1~C4）

| # | 约束 | 值 | 来源 | 会约束住什么 |
|---|---|---|---|---|
| C1 | 参数预算 | **450~550M** | 项目目标 | 所有维度的"总盘子"，加起来不能超 |
| C2 | 训练硬件 | 2× RTX 4090（24G×2），bf16 全参 DDP | 00 侦察 | 深度/宽度上限 + 显存账 |
| C3 | 词表 | **6400（锁死）** | 01 训好的 BPE，语料已编码 | embedding 支出 = 6400×hidden，改词表 = 01 白学 |
| C4 | 上下文 | **32768** | Qwen3 风格 | 长上下文用 RoPE，零参数成本 |

> 规矩：**任何设计决定违反 C1~C4 即无效**。

---

## 3. 组件决策（六个"为什么"）

每项都按"选了什么 → 为什么 → 落选方案 → 数字影响"讲。

### 3.1 注意力 → GQA（16 Q / 8 K + 8 V）

三个变体（MHA/GQA/MQA）**唯一区别在 KV 头数**（Q 头固定 16）：
- **Q 是"一次性"的**：decode 每步只算当前 token 的 Q，历史 Q 不存；
  **K/V 是"累积"的**：历史每个 token 的 K/V 必须缓存（KV cache，随 seq 线性增长）。
- KV cache（bf16，32k 上下文，24 层）：`2×24层×KV头×80×2B×seq`

| | 每 token | 32k 单条 | ×8 并发 |
|---|---|---|---|
| MHA (16) | 120 KB | 3.75 G | 30 G ← 24G 卡装不下 |
| **GQA (8)** | 60 KB | **1.88 G** | 15 G ✅ |
| MQA (1) | 7.5 KB | 234 M | 1.9 G |

- decode 是**带宽瓶颈**（每步读整个 KV cache）→ cache 减半 = 生成提速。
- **训练侧几乎无感**：attention 计算量由 16 个 Q 头决定，三变体相同；K/V 投影权重
  MHA 78.6M vs GQA 39.3M，GQA 只省 39M（~8%），投回别处。
- **质量**：MHA ≥ GQA > MQA；GQA(g=8) 基本追回 MHA（Ainslie 2023），LLaMA2/3、Qwen 全系 GQA。
- ⚠️ 术语坑：是 **8 个 K + 8 个 V**（K=索引、V=索引卡片上的内容，必须按头成对），不是"8 个 V"。

### 3.2 位置编码 → RoPE（base 1e6）

- **为什么必须**：纯 attention 打分是置换不变的（打乱词序，输出只是同样打乱）——不注入位置就学不到词序。
- **做法**：每个 Q/K 头 80 维拆成 **40 对二维平面**，第 i 对转 `pos × θᵢ`，θᵢ = base^(-2i/80)。
- **为什么是"相对"**：旋转矩阵正交 →
  `⟨R_m·q, R_n·k⟩ = ⟨q, R_{n−m}·k⟩`，绝对位置全抵消，**只剩差 n−m**（"看两根钟针的夹角"类比）。
- **三个推论**：零参数（旋转是公式非权重）｜32k 稳（base 1e6 最慢档周期 ~440 万位置）｜
  **Q 和 K 必须一起转**（都转绝对角才抵消），**V 不转**（只参与内容取回）。
- 落选：绝对位置 embedding（训练长度是上限）、ALiBi（固定惩罚不可学）。
- base 用 **1e6**（对齐 minimind 源码 `precompute_freqs_cis`/Qwen3）而非 1e4（最慢档周期仅 ~5 万，对 32k 偏紧）。

### 3.3 归一化 → RMSNorm（pre-norm）

- **RMSNorm** = 除以自身均方根（音量旋钮）+ 可学习逐维缩放（均衡器），只有 weight 无 bias。
- vs **LayerNorm**：LN 干两步（①减均值 ②除标准差）+bias；RMSNorm 只干②——
  **"少的一步"= 不减均值**，质量无损（Zhang & Sennrich 2019）、更快。
- **pre-norm**（`x = x + Attention(RMSNorm(x))`）：残差高速公路保持**干净恒等**，
  梯度走 `∂x/∂x = 1` 直通不回折；post-norm 把 norm 压在残差上，梯度要穿 24 次 norm → 难训
  （原始 Transformer 需重 warmup 的教训，LLaMA 起全线 pre-norm）。
  顶部 final RMSNorm 是唯一"尾部归一"，只在出最后一层时做一次。
- 全模型 97 个 RMSNorm ≈ 66.5K 参数。

### 3.4 FFN → SwiGLU（silu 门控，3 张矩阵）

```
GELU 旧路线：down( gelu( up(x) ) )            2 矩阵，每条中间神经元"全开"
SwiGLU 现役：down( silu( gate(x) ) · up(x) )  3 矩阵，gate 决定每条"开多大"
```
- 门控 = attention 软门的 FFN 版（逐元素闸门）；同等预算下质量更好（Shazeer 2020），LLaMA/Qwen 全系标配。
- 代价：多一张矩阵。FFN 汇集了 **74.6% 的参数量**当"知识库"，attention 只当"混音器"，这笔账值。
- I = 4032 的来历（minimind 源码同款公式）：`I = ceil(H·π/64)·64 = ceil(1280×3.1416/64)×64 = 63×64`，π 是配比系数。

### 3.5 三个开关（共享 & 训练技巧）

| 开关 | 值 | 为什么 |
|---|---|---|
| tie_word_embeddings | **true** | embed 与 lm_head 是**同一张 tensor**：省 8.19M + 输出与输入同空间天然正则 |
| dropout | **0.0** | 0.5B 本来就欠训练，dropout 只拖慢收敛；正则留给 04/05 后训练 |
| bias / flash_attn | **False / True** | 现代标配去 bias；torch SDPA 提速 |

### 3.6 附加：QK-Norm

- `q_norm / k_norm` = RMSNorm(head_dim=80)，在 **RoPE 之前**作用在 Q/K 上（minimind `model_minimind.py` L117 坐实），每层仅 160 参数。

---

## 4. 容量分配（为什么是 1280×24×4032）

**参数公式**（r = Q头/KV头 = 2）：每层 attention `A = 3H²`；每层 FFN `F = 3HI`；embedding `E = V·H`。

- **配比**：attn:ffn = F/A = I/H = 4032/1280 ≈ **1:3.15**——"attention 混音、FFN 知识"的钱怎么分，
  是 LLaMA/Qwen 祖传比例（minimind 的 π 系数出处）。FFN 是知识库拿大头。
- **三套候选裁决**（都填满 ~500M）：

| 方案 | H×L×I | attn:ffn | 总量 | 裁决 |
|---|---|---|---|---|
| A 宽而浅 | 1536×18×4416 | 1:2.9 | ~504M | attention 的 H² 吸走预算，深度只剩 18 步，长链路撑不住 ❌ |
| **B 均衡 ✅** | **1280×24×4032** | **1:3.15** | **497.8M** | 深度足 + 配比健康，0.5B 档最常用值 |
| C 深而窄 | 1024×32×4032 | 1:3.9 | ~504M | 32 步但注意力仅 1:3.9，交互是瓶颈 ❌ |

- **层数为什么是 24**：20 层=400M 浪费预算；26 层=538M 贴上限无余量；24 层=489.6M+8.2M≈497.8M，
  又是 0.5B 档最少出现问题的值（Qwen2.5/3-0.5B 同为 24 层）。
- **防线**（为什么不再宽/窄/深/多）：H≥1536 → 知识库被 H² 挤牙膏；H≤768 → 每层 attention 仅 1.8M 带宽不足；
  头数 32 → head_dim=40 太细；词表更小 → C3 锁死不重训。

---

## 5. 结构全景（零件与顺序）

```
input_ids [B,S]（token 编号）
  │
  ▼
① Embedding：查表 6400×1280 → [B,S,1280]          ← 数据流第一站（与 ④ 共享权重）
  │
  ▼
② 24 × Block：
  │  x = x + Attention(RMSNorm(x))                  ← attn 子层（在前）
  │       └ q/k/v_proj → q/k_norm(QK-Norm) → RoPE → SDPA → o_proj
  │  x = x + MLP(RMSNorm(x))                        ← MLP 子层（在后）
  │       └ silu(gate(x))·up(x) → down(x)（SwiGLU）
  │
  ▼
③ final RMSNorm
  │
  ▼
④ lm_head：1280→6400（与 ① 同一张权重）→ logits [B,S,6400]
```

术语：**"层"（num_hidden_layers=24）= 一个 Block**（attn 子层 + mlp 子层）；
Block 内 attn 在前、mlp 在后 = 原始 Transformer 顺序；pre-norm 保证残差"干净恒等"。

**参数矩阵地图**（每层重复 ×24；`[行×列]`，参数 = 行×列）：

```
embed_tokens（与 lm_head 共享）  [6400 × 1280]   8.19M         ← 词表查表
  Block:
    input_layernorm              [1280]          1,280
    q_proj                       [1280 × 1280]   1.64M  16 头×80
    k_proj                       [1280 × 640]    0.82M   8 头×80（GQA 共享）
    v_proj                       [1280 × 640]    0.82M   8 头×80
    q_norm / k_norm              [80] × 2        160    （RoPE 之前）
    o_proj                       [1280 × 1280]   1.64M  16 头输出合并
    post_attention_layernorm     [1280]          1,280
    gate_proj（SwiGLU 门控）      [1280 × 4032]   5.16M
    up_proj（内容通道）           [1280 × 4032]   5.16M
    down_proj（缩回）             [4032 × 1280]   5.16M
final RMSNorm                    [1280]          1,280
lm_head（= W_E）                 [1280 × 6400]   0      ← tie 共享，不占新参数
──────────────────────────────────────────────────
合计 497,812,480 ≈ 497.8M
```

> 比值记忆：每层 1280→4032→1280 是"先宽后窄"（SwiGLU 知识扩张再压缩）；
> FFN : attention = 15.48M : 4.92M ≈ 3.15 : 1（L2 的配比，π 的出处）。

---

## 6. 参数量审计（E2 实测与手工账一致）

| 组件 | 参数 | 占比 |
|---|---|---|
| Embedding（与 lm_head 共享，只算一次） | 8,192,000 | 1.6% |
| 24 × Attn（q/k/v/o_proj + q_k_norm） | 117,968,640 | 23.7% |
| 24 × FFN（gate/up/down） | 371,589,120 | 74.6% |
| RMSNorm（块内 48 + final 1 + qk 48） | 66,560 | 0.0% |
| **总计** | **497,812,480 ≈ 497.8M** | 100% |

- 单块账：attn 4.92M + ffn 15.48M + 2×norm 2,560 = 20.4M；24 层 = 489.6M。
- E2 用 `named_parameters()`（tied 去重，用 state_dict 会把共享权重数两遍）实测**偏差 0.000%**；
  相对 500M 预算 **99.56%**，C1 ✅。

---

## 7. 关键账

| 项 | 数 | 备注 |
|---|---|---|
| 训练计算量 | 6·N·T ≈ **4.5e18 FLOPs** | T = 1.5B token（首版） |
| 训练时长 | **15~28 h** | 2×4090 bf16 DDP |
| 训练显存 | 权重系 ~8G/卡 + 激活 | 24G×2 宽裕（bf16 全参） |
| 推理 KV cache | **1.88G/条 @32k** | GQA-8（若 MHA 将 3.75G） |
| 参数/预算 | 99.56% | C1 通过 |

---

## 8. 验证与风险

| 实验 | 结果 |
|---|---|
| E1 前向走查 | shape 流 (2,32)→(2,32,6400) 正确；初始 loss **9.06** ≈ ln(6400)=8.76（±0.3 = 随机初始化 logits 非全零的正常波动）；backward 无 NaN |
| E2 参数审计 | 497,812,480，手工账 vs 实例化偏差 **0.000%** |
| E3 最小 decoder | 110,912 参数纯 torch 从 0 学"后继规则"，loss 4.72 → **0.026** |

- **已知欠训练**：497.8M 配 1.5B token ≈ 3× Chinchilla 最优（~150M）。有意为之：
  先"会说话 + 按格式"，跑通后可选吃满 03 全量语料（4~5B token）补训一轮，**不动架构**。
- SFT/RL 显存吃紧的对策：seq 降到 1024 或开 gradient checkpointing，**不动架构**。

---

## 9. 定稿配置（生成/训练直接可用）

config_zzmind0.5b.json：

```json
{
  "_model": "zzmind-0.5B",
  "_version": "v1.0-final (2026-09-28 定稿)",
  "_task": "minicode：人格陪伴 + 2~4 工具调用 + 轻量推理 + 多意图路由",
  "_design_ref": "02_architecture/DESIGN.md",
  "model_type": "minimind",
  "hidden_size": 1280,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 80,
  "intermediate_size": 4032,
  "vocab_size": 6400,
  "max_position_embeddings": 32768,
  "tie_word_embeddings": true,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-6,
  "hidden_act": "silu",
  "use_moe": false,
  "num_experts": 4,
  "num_experts_per_tok": 1,
  "dropout": 0.0,
  "flash_attn": true
}
```

---

## 10. 变更记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-09-28 | 定稿；与旧工作坊"均衡方案 B"逐项一致（独立推导收敛到同一解 = 交叉验证设计正确性）|