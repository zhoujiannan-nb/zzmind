#!/bin/bash
# ============================================================
# zzmind-0.5B 预训练启动（node05，2×4090 DDP）
#   数据：主语料 7.8G + 路由语料 65.5 万条（逗号分隔混训，随机 shuffle）
#   预算：1.5e9 token ≈ 全量 73% → 183,105 步（每步 8×512×2卡 = 8192 token）
#   时长：单步 ~0.3~0.6s → 约 15~30 小时
#   存档：每 2000 步落 pretrain.pth（推理用）+ pretrain_resume.pth（续训用）
# 用法：
#   bash start_pretrain.sh            # 从头训
#   bash start_pretrain.sh resume     # 崩溃/中断后续训（自动跳过已训步数）
# ============================================================
set -e
cd /mnt/boot/datasets/zzmind/03_pretrain

RESUME_FLAG=""
[ "$1" = "resume" ] && RESUME_FLAG="--from_resume 1"

torchrun --nproc_per_node=2 trainer/train_pretrain.py \
  --data_path /mnt/boot/datasets/zzmind/pretrain_t2t.jsonl,/mnt/boot/datasets/zzmind/router_for_pretrain.jsonl \
  --save_dir /mnt/boot/datasets/zzmind/out \
  --save_weight pretrain \
  --epochs 1 \
  --batch_size 8 \
  --max_seq_len 512 \
  --accumulation_steps 4 \
  --learning_rate 5e-4 \
  --warmup_steps 1000 \
  --total_tokens 1.5e9 \
  --log_interval 50 \
  --save_interval 2000 \
  --num_workers 8 \
  $RESUME_FLAG \
  2>&1 | tee /mnt/boot/datasets/zzmind/out/pretrain_$(date +%m%d_%H%M).log
echo "pretrain done: $(date)"
