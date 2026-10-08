# 03 · 预训练：在中文语料上从零学语言

## 目标
用主语料（7.8G 中文）+ 路由语料（65.5 万条自合成）把 0.5B 模型从随机初始化
训到"能写出通顺中文"。这是整个旅程最核心的一步——模型 90% 的语言能力在这里形成。

## 核心概念
1. **训练循环**：`数据 → 输入 [B,N] token → logits → CE loss → 自动反向传播（torch 记计算图倒着走）→ AdamW 更新`。
   一行 `loss.backward()` 把梯度传回每个参数——你说的"倒着走一遍"就是这个。
2. **任务**：next-token prediction + cross-entropy loss。
   随机模型初始 loss ≈ ln(6400) ≈ 8.76（均匀分布的熵），训得好才会下降。
3. **学习率调度**：warmup（前 N 步从 0 升到峰值，防早期震荡）+ cosine 衰减（慢慢降到峰值的 10%）。
   `trainer_utils.get_lr()` 实现。
4. **bf16 混合精度**：前向/反向用半精度（速度快 2 倍、显存省一半），权重本体 fp32。
   bf16 指数位和 fp32 一样宽不会溢出，所以免 GradScaler（fp16 才会溢出才需要）。
5. **梯度累积**：显存装不下大 batch 时，先攒 N 个小 batch 的梯度再一起更新。
   有效 batch（token/优化器更新）= micro_batch × accum_steps × seq_len × GPU 数。
6. **DDP 数据并行**：2 张卡各拿一半数据，每步反向后 all-reduce 同步梯度（"双卡怎么交互"的答案）。
   初始权重：每卡各自随机初始化后，DDP 构造时把 rank0 的参数广播给所有卡对齐。
7. **梯度裁剪**：clip_grad_norm_(params, 1.0)，防大梯度一步把模型冲坏。
8. **数据流式访问**：PretrainDataset 建"行偏移表"，__getitem__ 才 seek 读那一行——
   7.8G 语料不整包载入内存，开机即用。

## 代码地图
| 文件 | 作用 |
|---|---|
| `start_pretrain.sh` | 正式训练启动（node05 双卡；`resume` 参数续训） |
| `model.py` | 02 定稿的 0.5B 模型（497.8M）+ next-token loss 计算 + generate |
| `config_zzmind0.5b.json` | 定稿配置落盘（模型侧参数） |
| `dataset.py` | PretrainDataset：jsonl 行偏移随机访问 + 截断 + pad 位屏蔽（-100）；**逗号分隔多文件混训** |
| `trainer/train_pretrain.py` | 主入口：DDP/bf16/累积/clip/cosine/存档/续训/token 预算到步停 |
| `trainer/trainer_utils.py` | get_lr / DDP 初始化 / 种子 / checkpoint / 模型+tokenizer 装载 |
| `token_rate.py` | token/字符 比率实测工具（数据侦察用，不在训练链路里） |
| `merge_router.py` | 路由语料并入脚本（已跑完，产物 `/mnt/boot/datasets/zzmind/router_for_pretrain.jsonl`） |
| `model/` | 01 阶段 6400 词表（tokenizer.json + config），训练直接用 |

## 数据与预算
- 主语料 `pretrain_t2t.jsonl`：7.8G / 8,468,827 行 ≈ **1.76B token**
  （实测 token/字符 = 0.758；平均 274 字符/行 ≈ 208 token/行，99.8% 行 ≤1023 token）
- 路由语料 `router_for_pretrain.jsonl`：655,176 行 ≈ **0.305B token**（~466 token/行，data_aug 产出）
- **混训**：`--data_path 主语料.jsonl,路由.jsonl` 逗号分隔，DistributedSampler 随机 shuffle，
  路由语料约占训练 token 的 ~15%（行数占比 7%，但单条 token 密度高）
- **预算 1.5e9 token ≈ 全量的 73%**，到步自动停，不硬吃完语料

## 训练配置（定稿）
| 项 | 值 | 备注 |
|---|---|---|
| 超参 | 1280 / 24层 / GQA 16:8 / 4032 / 6400 / 32k | config json 即定稿值 |
| seq_len | 512 | 跑稳后可试 1024 |
| micro_batch/GPU | 8 | bf16 权重 ~4G/卡 + 激活，24G 宽裕 |
| accum_steps | 4 | 有效 batch = 8×4×512×2卡 = 32,768 token/优化器更新 |
| lr | 5e-4（cosine → 5e-5） | warmup 1000 步 |
| **总步数** | 1.5e9 ÷ 8,192 ≈ **183,105 步** | 每步(微批) = 8×512×2卡 = 8,192 token，到步自动停 |
| 精度 / 硬件 | bf16 / 2×4090 DDP | 免 GradScaler |
| 保存 | 每 2000 步存 ckpt + resume | 崩溃后续训不白练 |

**时长估算**：单步(微批) ~0.3~0.6s → 183,105 步 ≈ **15~30 小时**。
开跑后以日志实测 tok/s 为准修正 ETA。

## 怎么跑
```bash
# node05 双卡正式
cd /mnt/boot/datasets/zzmind/03_pretrain
bash start_pretrain.sh            # 从头训
bash start_pretrain.sh resume     # 崩溃/中断后续训（自动从 pretrain_resume.pth 跳过已训步数）

# 本地 CPU 冒烟（改配置前先验证链路，小文件）
python 03_pretrain/trainer/train_pretrain.py --data_path 小文件.jsonl \
    --device cpu --epochs 1 --batch_size 2 --max_seq_len 128 \
    --accumulation_steps 1 --warmup_steps 0 --total_tokens 100000 \
    --num_workers 0 --log_interval 1
```

**监控**：
- 日志 `out/pretrain_MMDD_HHMM.log`：每 50 步一行 loss / lr / tok/s / ETA
- 权重 `out/pretrain.pth`（fp16，推理/generate 用）+ `out/pretrain_resume.pth`（含 optimizer 状态，续训用）
- GPU：`nvidia-smi`（显存 / 利用率）
- 曲线（可选）：启动参数加 `--use_swanlab`（需 node05 已装 swanlab）

**流程**：
1. 启动后前 100 步盯三件事：loss 从 ~8.76 平滑下降（无跳变 spike）、显存不 OOM、tok/s 稳定
2. 中途可 kill / 断电，`bash start_pretrain.sh resume` 续训（存档点均为累积整数倍，步数对齐）
3. 训练期间定期用 checkpoint generate 几段文本存档，看模型什么时候开始说人话

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

## 验收标准
1. loss 降到 < 3.5，generate 能出通顺的中文段落（贴 3 段到 notes）
2. 学习清单能独立讲清
3. 有完整 loss 曲线 + 人评样本记录
4. 通过后：主 README 状态改 ✅，进入 04_sft

## 下一步
04_sft：模型会"写"了，但不会"对话"——教它认 chat template、学会答话。
