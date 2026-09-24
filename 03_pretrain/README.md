# 03 · 预训练：在 7.8G 中文语料上学语言

## 目标
用 pretrain_t2t.jsonl 把 0.5B 模型从随机初始化训到"能写出通顺中文"。
这是整个旅程里最核心的一步——模型 90% 的语言能力在这里形成。

## 核心概念
1. **任务**：next-token prediction + cross-entropy loss。
   随机模型初始 loss ≈ ln(6400) ≈ 8.76（均匀分布的熵），训得好才会下降。
2. **学习率调度**：warmup（前几千步从 0 升到峰值，防早期震荡）
   + cosine 衰减（慢慢降到峰值的 10%）。minimind 的 get_lr() 就是这套。
3. **bf16 混合精度**：前向/反向用半精度（速度快 2 倍、显存省一半），
   但梯度累加和 master 权重用 fp32。bf16 比 fp16 指数位多，不容易溢出。
4. **梯度累积**：显存装不下大 batch 时，先攒 N 个小 batch 的梯度再更新。
   有效 batch = micro_batch × accum_steps × GPU 数。
5. **DDP 数据并行**：2 张卡各拿一半数据，每步后 all-reduce 同步梯度。
   代码里 init_distributed_mode() + DistributedDataParallel。
6. **梯度裁剪**：clip_grad_norm_(params, 1.0)，防大梯度一步把模型冲坏。
7. **数据 packing**：多条短文本拼接成一个 seq 提高利用率。
   拼接处要处理"跨样本注意力污染"——minimind 用特殊 buffer token 分隔
   （01 阶段见过的 <|buffer1|>~<|buffer9|> 就是干这个的）。

## 数据事实（00 阶段侦察结果）
- pretrain_t2t.jsonl：7.8G / 8,468,827 行 / {"text": "..."}，中文为主
- 平均 ~920 字节/行 ≈ 600 字 ≈ 400~600 token（6400 词表，待 E4 实测）
- 全量 ≈ **4~5B token**，过一遍数据就是 4~5B token 的训练量
- 我们第一版只训 ~1.5B token（约 1/3 数据），先要"能说话"，别贪多

## 0.5B 预训练配置（拟定，E1 实测后定稿）
| 项 | 值 |
|---|---|
| 超参 | 02 阶段表格（1280 / 24层 / GQA 16:8） |
| seq_len | 512（跑稳后可试 1024） |
| micro_batch/GPU | 8~16（E1 实测显存后定） |
| accum_steps | 4~8 |
| 有效 batch | 64~128 |
| lr | 5e-4 → 5e-5（cosine） |
| warmup | ~1000 步 |
| 总步数 | ~100k 步（≈1.5B token） |
| 精度 | bf16 |
| 硬件 | 2×4090 DDP |
| 保存 | 每 2000 步存 ckpt，按 loss 保留 best |

**时间估算**：0.5B、seq 512、有效 batch ~128，单步约 0.5~1s →
100k 步 ≈ **15~28 小时**。适合开跑后睡觉，第二天看曲线。

## 学习清单
- [ ] 为什么初始 loss 恰好是 ln(vocab_size)
- [ ] loss 和困惑度（perplexity）什么关系，怎么读 loss 曲线
- [ ] bf16 和 fp16 的区别，为什么选 bf16
- [ ] 梯度累积怎么等效大 batch（推一遍）
- [ ] DDP 里"有效 batch"怎么算，什么时候需要梯度累积
- [ ] packing 的跨样本污染是什么，buffer token 怎么解决
- [ ] 学习率不调会怎样（一直大 / 一直小）
- [ ] loss spike 了怎么办（排查清单：lr 太高？坏数据？硬件？）

## 实验任务
- **E1 mini 跑**：2000 步（约 1 小时）。验收：
  显存占用、训练速度（it/s）、loss 从 8.76 稳定降到 ~5 左右。
- **E2 正式跑**：100k 步。每 5000 步用当前权重 generate 5 段
  "人评样本"存到 samples/（看模型什么时候开始说人话）。
- **E3 曲线分析**：wandb/swanlab 或自己画图，标注 spike，写分析到 notes。
- **E4（01 遗留）**：实测 6400 词表的 token/字符 比率。

## 验收标准
1. loss 降到 < 3.5，generate 能出通顺的中文段落（贴 3 段到 notes）
2. 学习清单 8 条能独立讲清
3. 有完整 loss 曲线 + 人评样本记录
4. 通过后：主 README 状态改 ✅，进入 04_sft

## 下一步
04_sft：模型会"写"了，但不会"对话"——教它认 chat template、学会答话。
