# 02 · 模型架构：把 0.5B 拆成零件看

## 目标
训练之前，把模型结构完全看懂：
- 每个组件是什么、为什么这么设计
- 能手算 0.5B 的参数量
- 能亲手写一个 2 层迷你 GPT 验证理解

## 整体结构（一次前向传播）

```
input_ids [B, S]   （B=batch, S=序列长）
   │
   ▼
Embedding(6400, 1280)          token id → 向量
   │
   ▼
   ┌──────────────────────────────────────────────┐
   │  × 24 层 MiniMindBlock：                     │
   │     x = x + Attention(RMSNorm(x))   ← 自注意力 │
   │     x = x + SwiGLU(RMSNorm(x))        ← 前馈  │
   └──────────────────────────────────────────────┘
   │
   ▼
RMSNorm（final norm）
   ▼
lm_head: Linear(1280, 6400)    ← 与 Embedding 共享权重(tied)
   │
   ▼
logits [B, S, 6400]
   ├─ 训练：与"下一个 token"算 cross-entropy loss
   └─ 推理：softmax 后采样出下一个 token
```

一句话：输入一段文字，模型在**每个位置**预测"下一个 token 的概率分布"，
训练就是让分布对准真实下一个 token，如此反复。

## 组件拆解（7 个零件）

### 1. Embedding（词嵌入）
token id（0~6399 的整数）→ 1280 维向量。参数 6400×1280 = 8.2M。
"共享权重"（tie）：lm_head 的矩阵 = Embedding 的转置，省 8.2M 参数，
0.5B 这种小模型下效果几乎无损（大模型才分开）。

### 2. RMSNorm（均方根归一化）
`x * rsqrt(mean(x²) + eps) * weight`。只缩放不归零（对比 LayerNorm 少了减均值）。
作用：把每维向量的"幅度"压回正常范围，防止深层网络里数值爆炸/消失。
参数极少（每层 2×1280）。Qwen3/LLaMA 全系标配。

### 3. RoPE（旋转位置编码）
不额外加位置向量，而是把 q、k 向量**按位置旋转一个角度**：
位置 i 的向量转 θ·i 度。两个位置的点积只依赖它们的**相对距离** i-j，
这就是"相对位置"。θ 由 base=1e6 的几何级数决定（低频学长距离，高频学近距离）。
长上下文用 YaRN 缩放（config 里的 rope_scaling），0.5B 阶段先不开。

### 4. Attention（GQA + QK-Norm + causal mask）
- **causal mask**：第 i 个位置只能看 0~i（不能偷看未来），这是"自回归"的来源。
- **MHA → GQA**：16 个 Q 头两两共享 1 组 K/V 头（8 组）。
  推理时 KV cache 减半（0.5B 聊天/agent 多轮对话省显存的关键）。
- **QK-Norm**：点积前对 q、k 各做一次 RMSNorm。
  解决"训着训着 q·k 幅度漂移 → softmax 饱和 → 梯度消失"的问题（Qwen3 技巧）。
- 复杂度 O(S²·d)：S=2048 时每 token 要算 2048 个点积，这就是序列不能无限长的原因。

### 5. SwiGLU FFN（前馈网络）
三层矩阵：gate、up、down，中间是 `SiLU(gate(x)) * up(x)`，再 down 回 1280。
为什么 3 个矩阵而不是 2 个（传统 FFN 只有 W1/W2）：GLU 门控让网络能
"选择性激活"，同参数量下拟合能力更强（Llama 论文结论）。
每层参数 3×1280×4032 ≈ 15.5M，**每层里 FFN 占 3/4 参数**——
大模型的"知识"主要存在 FFN 里，注意力主要学"关系/路由"。

### 6. 残差连接
`x = x + Attention(...)`、`x = x + FFN(...)`。梯度可以沿残差通路直通底层，
没有它，24 层网络根本训不动。

### 7. generate（采样解码）
每步取最后一个位置的 logits，依次做：
temperature 缩放 → top_k 截断（只留前 50 个）→ top_p 截断（累积概率 0.85）
→ softmax 多项式采样 → 遇到 eos_token_id 停止。
温度/top_p 越高越"有个性"（随机），越低越"规矩"（确定性）——05 阶段调性格要用。
## 0.5B 超参数与参数手算

| 超参数 | 值 | 推导 |
|---|---|---|
| hidden_size | 1280 | 经验值（1280 = 16 头 × 80） |
| num_hidden_layers | 24 | 目标总参数 ~0.5B 反推 |
| num_attention_heads | 16 | |
| num_key_value_heads | 8 | GQA：2 个 Q 头共享 1 组 KV |
| head_dim | 80 | 1280 / 16 |
| intermediate_size | 4032 | minimind 公式 ceil(h·π/64)·64 |
| vocab_size | 6400 | 01 阶段的词表 |
| max_position_embeddings | 32768 | RoPE 表长度 |
| tie_word_embeddings | true | 小模型省参数 |

**每层参数手算（你来做 E2 前先在纸上算一遍）：**
- 注意力：q 1280×1280 + k 1280×640 + v 1280×640 + o 1280×1280 = 4.92M
- FFN：3 × 1280 × 4032 = 15.47M
- norm：2 × 1280 ≈ 0
- 每层合计 ≈ **20.4M** → × 24 层 = 489.6M
- Embedding：6400 × 1280 = 8.2M（lm_head 共享，不重复计）
- **总计 ≈ 498M ✓ 0.5B**

## 可视化（先读这个）
- **`architecture_viz.html`**：单文件页面（图内嵌，浏览器直接打开），四张图 + 讲解：
  ① 模型 = 8 种 Linear 层 + 4 个"三明治" ② 前向传播张量形状走查
  ③ 参数审计 ④ 为什么是这个形状（2017→2026 演化）
- **`figures/`**：四张图的 PNG 版（fig1~fig4），需要插图时用

## 代码地图（minimind/model/model_minimind.py）
- L10-46  `MiniMindConfig`：所有超参数 + YaRN 配置
- L50-60  `RMSNorm`：归一化
- L62-78  `precompute_freqs_cis`：RoPE 旋转表预计算
- L80-134 `Attention`：L117 QK-Norm、L124 GQA repeat_kv、L125-131 flash attention 分支
- L136-146 `FeedForward`：SwiGLU 三矩阵
- L148-181 `MOEFeedForward`：专家路由（03 阶段可选实验）
- L183-199 `MiniMindBlock`：残差块
- L201-237 `MiniMindModel`：主干循环 + 位置编码切片
- L239-258 `MiniMindForCausalLM`：forward + loss（注意 L256 labels 右移一位）
- L262-293 `generate`：采样解码全流程

## 学习清单（验收时逐条回答）
- [ ] 用大白话解释"下一个 token 预测"
- [ ] 注意力为什么是 O(S²)？seq=2048 时具体是多少次点积
- [ ] GQA 比 MHA 每层省多少 KV 参数？推理时省多少 KV cache
- [ ] RoPE 为什么用旋转？点积里"相对位置"是怎么出来的
- [ ] QK-Norm 不加会出什么事
- [ ] SwiGLU 为什么是 3 个矩阵
- [ ] tie embedding 是什么、为什么小模型要开
- [ ] 手算 0.5B 总参数量，与 E2 输出对比（误差 < 5%）
- [ ] 采样管线各步骤顺序与各自作用

## 实验任务（脚本在 `experiments/`，node05 或本机 CPU 都能跑）
- **E1 前向走查**（`experiments/E1_forward_walk.py`）：随机初始化 0.5B，喂一句话，打印每一层输出形状、
  logits 范围、初始 loss（应 ≈ ln(6400) ≈ 8.76）。验证你对形状流的理解。
  2026-09-25 已在 node05 跑通（GPU 被 SGLang 占用时走 CPU 也行）。
- **E2 参数审计**（`experiments/E2_param_audit.py`）：统计每个模块参数量，打印占比，对比手算值。
  2026-09-25 已跑通：实测 497.812M，与手算偏差 0.01%。
- **E3 手写迷你 GPT**（`experiments/E3_mini_gpt.py`，自包含只要 torch）：不看 model_minimind.py，用 ~60 行 PyTorch 写一个
  2 层、64 维、小词表的 GPT，在 CPU 上训练"固定模式"（a→b→a→b），
  loss 降下来并能正确预测下一个 token 即算通过。2026-09-25 已跑通（~30 秒）。
- **E4 注意力可视化（选做）**：取 E1 的模型，画某一层对某句话的
  注意力矩阵热力图，观察 causal 下三角结构。

## 验收标准
1. 学习清单 9 条全部能回答（在 notes.md 里写答案）
2. E1/E2 实际输出与手算一致
3. E3 迷你 GPT 训练曲线下降且预测正确
4. 通过后：主 README 状态列改 ✅，进入 03_pretrain

## 记录区
理解、问题、实验输出都写到 [notes.md](notes.md)。

---

## ★ 架构设计工作坊（自己设计，不是抄 minimind）

上面是"读懂零件"。这一节是**我们自己的 0.5B 要长什么样**——
每个决策都摆出选项和权衡，你来拍板，最终落成
`design/config_zzmind0.5b.json` + `design/DESIGN.md`（记录你为什么这么选）。
**能讲清"为什么选 A 不选 B"才算设计，抄数字不算。**

### 决策 1：模型形状（最核心，决定其余全部）
参数量 ≈ 12·L·h² 量级（含 FFN）。目标 ~0.5B：
| 方案 | hidden × 层数 | 头(head_dim) | FFN 宽 | 特点 |
|---|---|---|---|---|
| 瘦高 | 768 × 40 | 12(64) | 2432 | 层多→长程依赖强，单头小 |
| 均衡(推荐) | 1280 × 24 | 16(80) | 4032 | 对齐 Qwen3-0.6B 思路，稳 |
| 矮胖 | 2048 × 14 | 16(128) | 6528 | 层少→快，单头大，长程弱 |
- 权衡：层数多 = 更多"思考步数"、长上下文好，但训练慢、易震荡；
  hidden 大 = 单 token 表达力强，但每层参数暴涨（h²）。
- **你选：____**（推荐"均衡"，想省时间可"矮胖"）

### 决策 2：注意力方案
| 方案 | Q头 : KV头 | 效果 |
|---|---|---|
| MHA | 16:16 | 质量上限，KV cache 最大 |
| GQA(推荐) | 16:8 | 质量≈MHA，KV cache 减半（06 多轮 agent 省显存关键） |
| MQA | 16:1 | 最快，质量掉一点 |
- **你选：____**（推荐 GQA 16:8）

### 决策 3：FFN 宽度（intermediate_size）
- minimind 公式 `ceil(h·π/64)·64`（≈3.14h，SwiGLU 甜点位）。
- 选大：知识容量↑但参数↑；选小：省参数但"脑子变小"。
- **你选：____**（推荐用公式值；想冲 0.5B 上限可 +1 档）

### 决策 4：Dense vs MoE
| 方案 | 说明 |
|---|---|
| Dense(推荐) | 全参数激活，稳，先做这个 |
| MoE(加餐) | 4 专家 top1，激活参数≈1/4，推理快但路由要学 |
- **你选：____**（推荐先 Dense 出主线，MoE 当 03 的加餐实验）

### 决策 5：上下文长度 max_position_embeddings
- 8192 / 32768（推荐，开 YaRN）/ 131072。
- 权衡：RoPE 表 O(max_pos)，越大越占初始化内存；0.5B 聊天+agent 够 32k。
- **你选：____**（推荐 32768）

### 决策 6：其他小项（给推荐值，可改）
- rope_theta：1e6（默认）
- tie_word_embeddings：true（省 8.2M）
- rms_norm_eps：1e-6
- dropout：0.0（预训练不开，SFT 可选 0.05）
- flash_attn：true

### 设计流程
1. 你在 `design/DESIGN.md` 把决策 1~6 填了 + 写理由
2. 我据此生成/更新 `design/config_zzmind0.5b.json`
3. E1/E2 用**你的配置**跑（不是 minimind 默认），参数手算和实际对账
4. 一句话设计哲学写进 notes.md（将来 model card 的 tagline）
