# 02 · notes（学习记录）

格式：`[决定] / [理解] / [疑问] / [坑]` + 一句话理由。每层结束记一条小结。

## L0 约束确认（2026-09-28）
- [决定] 参数预算 = 0.5B（容差 450~550M）。理由：锚点是**迭代成本**——
  100M 撑不起人格+工具，7B 换一版人格按月上计，0.5B 一周能出一两版。
- [决定] 四条硬约束：C1 预算 450~550M / C2 2×4090 bf16 全参 DDP /
  C3 词表 6400 锁死（01 的 BPE 已把语料编码了）/ C4 上下文 32k（RoPE）。

## L1 骨架选择（2026-09-28）
- [已解决] KV 头 16→8 省了什么？训练：几乎不省（attention 计算量由 16 个 Q 头
  决定，三变体相同；K/V 投影权重只省 39M ~8%，L2 投回别处）。推理：KV cache 砍半
  （32k：3.75G→1.88G）——因为 Q 不缓存、K/V 逐 token 缓存，decode 是带宽瓶颈，
  cache 小 = 生成快。
- [决定] 注意力类型 = GQA：16 Q 头 / 8 K 头 / 8 V 头（K、V 必须成对等头数：
  K 是索引、V 是索引卡片上的内容，按头配对加权求和）。KV cache 按"KV 头"计。详见 §1
- [决定] 位置编码 = RoPE（base 1e6，max_pos 32768，零参数）：Q/K 按位置旋转，
  点积只依赖相对位置（绝对位置被旋转正交性消掉）；V 不旋转。base 对齐 minimind 源码。详见 §2
- [理解] 术语："层" = Block（attn 子层 + mlp 子层）；数据流第一站 = Embedding 不是 attention；
  Block 内 attn 在前、mlp 在后（原始 Transformer 顺序）。详见 L1 §7 全景图
- [发现] QK-Norm 的位置：q_norm/k_norm = RMSNorm(80)，在 RoPE 之前作用在 Q/K 上
- [粗算] 一个 Block ≈ 20.4M → 总计 497.8M，落在 C1 预算内（"为什么是这些数"留 L2）
- [理解] RMSNorm = 向量除以自身均方根（音量旋钮）+ 可学习均衡器；
  比 LayerNorm 少"减均值"一步、无 bias；全模型 97 个、~66.5K 参数。详见 L1 §3.1
- [理解] pre-norm 答案：残差直通高速"干净"，梯度走纯恒等路径（∂x/∂x=1）→ 24 层不衰减；
  post-norm 把 norm 压在残差上，梯度要穿 24 次 norm → 难训。详见 L1 §3.2
- [决定] FFN = SwiGLU（silu 门控、3 矩阵，知识库占块 76%）；tie = true（省 8.19M）；
  dropout=0、bias=False、flash_attn=True。详见 L1 §4~6

## L2 容量分配 + 定稿（2026-09-28）
- [定稿] v1.0 完整架构落定：**DESIGN.md + config_zzmind0.5b.json**（1280×24×4032，
  16Q/8KV GQA，RoPE 1e6，tie，总参 ≈497.8M）。
  与旧工作坊"均衡方案 B"完全一致 → 独立推导收敛到同一解，交叉验证设计正确。
- [理解] 配比公式：r=2 时 attn:ffn = I/H = 4032/1280 ≈ 1:3.15（LLaMA/Qwen 同族），
  FFN 是知识库拿大头、attention 是混音器。
- [验证] E1 forward：shape 正确；(2,32)->(2,32,6400)；初始 loss 9.06 ≈ ln(6400)=8.76
  （随机初始化 logits 非全零的 ±0.3 正常波动）；backward 梯度非 NaN
- [验证] E2 param audit：497,812,480 = 497.8M，手工账与实例化偏差 0.000%（相对 500M 99.56%，C1 通过）
- [验证] E3 mini GPT：110,912 参数纯 torch 最小 decoder，学"后继规则" loss 4.72 → 0.026

## 待办
- [ ] E3 mini GPT：无死角全部用普通 pytorch 重写一遍最小 decoder（理解零件在干嘛）
- [ ] 02 验收自检：能对 config 每个超参解释"为什么"（DESIGN.md 是对照答案）
- [ ] 通过后：主 README 状态改 ✅，进入 03_pretrain