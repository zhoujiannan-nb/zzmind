# 03 · 预训练：在 7.8G 中文语料上学语言

## 目标
用 pretrain_t2t.jsonl 把 0.5B 模型从随机初始化训到"能写出通顺中文"。
这是整个旅程里最核心的一步——模型 90% 的语言能力在这里形成。

## 核心概念
1. **训练循环**：`数据 → 输入 [B,N] token → logits → CE loss → 自动反向传播（torch 记计算图倒着走）→ AdamW 更新`。
   一行 `loss.backward()` 把梯度传回每个参数——你说的"倒着走一遍"就是这个。
2. **任务**：next-token prediction + cross-entropy loss。
   随机模型初始 loss ≈ ln(6400) ≈ 8.76（均匀分布的熵），训得好才会下降。
3. **学习率调度**：warmup（前 N 步从 0 升到峰值，防早期震荡）+ cosine 衰减（慢慢降到峰值的 10%）。
   `trainer_utils.get_lr()` 实现。
4. **bf16 混合精度**：前向/反向用半精度（速度快 2 倍、显存省一半），不溢出所以免 GradScaler。
5. **梯度累积**：显存装不下大 batch 时，先攒 N 个小 batch 的梯度再一起更新。
   有效 batch（token/步）= micro_batch × accum_steps × seq_len × GPU 数。
6. **DDP 数据并行**：2 张卡各拿一半数据，每步反向后 all-reduce 同步梯度（"双卡怎么交互"的答案）。
7. **梯度裁剪**：clip_grad_norm_(params, 1.0)，防大梯度一步把模型冲坏。
8. **数据流式访问**：PretrainDataset 建"行偏移表"，__getitem__ 才 seek 读那一行——
   7.8G 语料不整包载入内存，开机即用。

## 骨架文件（已落地，冒烟自检通过）
| 文件 | 作用 |
|---|---|
| `model.py` | 02 定稿的 0.5B 模型（497.8M，已完成） |
| `config_zzmind0.5b.json` | DESIGN §9 定稿配置落盘（模型侧参数） |
| `dataset.py` | PretrainDataset：jsonl 行偏移随机访问 + 截断 + pad 位屏蔽（-100） |
| `token_rate.py` | E4 实测 token/字符 比率与全量 token 估算（node05 上跑） |
| `trainer/train_pretrain.py` | 主入口：DDP/bf16/累积/clip/cosine/存档/续训 |
| `trainer/trainer_utils.py` | get_lr / DDP 初始化 / 种子 / checkpoint / 模型+tokenizer 装载 |
| `model/` | 01 阶段 6400 词表（tokenizer.json + config），已复制到此供训练用 |

## 数据事实（00 阶段侦察 + E4 实测修正）
- pretrain_t2t.jsonl：7.8G / 8,468,827 行 / {"text": "..."}，中文为主
- **E4 实测（2026-09-29, 抽样 2 万行）**：
  - token/字符 = **0.758**（汉字节中式：1 字符 ≈ 1.32 token）
  - 平均每行 **274 字符 / 208 token**；90.5% 行 ≤255 token，98.1% ≤511 token，99.8% ≤1023
  - 全量 ≈ **1.76B token**（早期估的 4~5B 偏高了——6400 小词表下 token 密度比想象低）
- **预算修正**：1.5B token ≈ 全量的 **85.2%**，即"第一版 ≈ 吃一遍语料的 85%"
  （原来"约 1/3 数据"的说法作废，训练器 `--total_tokens 1.5e9` 到步自动停仍成立）
- 路由语料（data_aug 产出，落满后再并入）：68 万条 × ~208 token ≈ **0.14B token**，占预算 ~9%

## 0.5B 预训练配置（拍板默认值，E1 实测后可调）
| 项 | 值 | 备注 |
|---|---|---|
| 超参 | 1280 / 24层 / GQA 16:8 / 4032 / 6400 // 32k | config json 即定稿值 |
| seq_len | **512** | 跑稳后可试 1024 |
| micro_batch/GPU | **8** | 24G 宽裕（bf16 权重系 ~8G/卡 + 激活）|
| accum_steps | **4** | 梯度累积补足 batch |
| 有效 batch | 8×4×512×2卡 = **32,768 token/步** | 你之前推的公式 |
| lr | 5e-4（cosine → 5e-5） | minimind 同款 |
| warmup | 1000 步 | ≈ 2% 总步数 |
| **总步数** | 1.5e9 / 32,768 ≈ **45,776 步** | 预算折算，到点自动停 |
| 精度 / 硬件 | bf16 / 2×4090 DDP | |
| 保存 | 每 2000 步存 ckpt + resume | 崩溃后续训不白练 |

**时间估算**：0.5B、seq 512、32,768 token/步，单步 ~1.3~2.2s →
45,776 步 ≈ **15~28 小时**。适合开跑后睡觉，第二天看曲线。

## 怎么跑
```bash
# 本地 CPU 冒烟（小文件，验证链路）
python 03_pretrain/trainer/train_pretrain.py --data_path 小文件.jsonl \
    --device cpu --epochs 1 --batch_size 2 --max_seq_len 128 \
    --accumulation_steps 1 --warmup_steps 0 --total_tokens 100000 \
    --num_workers 0 --log_interval 1

# node05 双卡正式（2×4090）
cd /mnt/boot/datasets/zzmind   # 代码传上去后的目录
torchrun --nproc_per_node=2 03_pretrain/trainer/train_pretrain.py \
    --data_path /mnt/boot/datasets/zzmind/pretrain_t2t.jsonl \
    --save_dir /mnt/boot/datasets/zzmind/out
```

## 学习清单
- [x] 训练 = 数据→前向→loss→自动反向→更新（torch autograd 记图倒走）
- [x] 双卡交互 = DDP：每卡同模型异数据，反向后 all-reduce 梯度求平均
- [x] 梯度累积等效大 batch（有效 batch = micro×accum×seq×GPU）
- [ ] 为什么初始 loss 恰好是 ln(vocab_size)，loss 和困惑度什么关系
- [ ] bf16 和 fp16 的区别，为什么选 bf16
- [ ] DDP 里"有效 batch"怎么算，什么时候需要梯度累积
- [ ] 学习率不调会怎样（一直大 / 一直小）
- [ ] loss spike 了怎么办（排查清单：lr 太高？坏数据？硬件？）
- [ ] packing 的跨样本污染是什么（优化项：buffer token 拼接，暂未做）

## 实验任务
- **E1 mini 跑**（node05，2000 步 ≈ 1 小时左右）。验收：
  显存占用、it/s、吞吐 tok/s、loss 从 8.76 稳定降到 ~5。
  顺手实测 6400 词表的 token/字符 比率（01 遗留问题）。
- **E2 正式跑**：45,776 步。每 5000 步用当前权重 generate 5 段
  "人评样本"存到 samples/（看模型什么时候开始说人话）。
- **E3 曲线分析**：swanlab/wandb 或自己画图，标注 spike，写分析到 notes。

## 验收标准
1. loss 降到 < 3.5，generate 能出通顺的中文段落（贴 3 段到 notes）
2. 学习清单 8 条能独立讲清
3. 有完整 loss 曲线 + 人评样本记录
4. 通过后：主 README 状态改 ✅，进入 04_sft

## 下一步
04_sft：模型会"写"了，但不会"对话"——教它认 chat template、学会答话。
