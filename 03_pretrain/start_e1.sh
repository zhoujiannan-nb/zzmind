#!/bin/bash
# ============================================================
# E1 冒烟启动脚本（node05，2×4090）—— GPU 空闲后一句跑
#   验证：显存占用 / 吞吐 tok/s / loss 从 8.76 稳定下降
#   时长：2000 步 × ~1.5~2s ≈ 1 小时
# 用法：bash /mnt/boot/datasets/zzmind/03_pretrain/start_e1.sh
# ============================================================
set -e
cd /mnt/boot/datasets/zzmind/03_pretrain

# ---- E1: 2000 步，只吃主语料一小部分 ----
# total_tokens = 2000 步 × 有效 batch(8×4×512×2=32768) ≈ 65.5M token
torchrun --nproc_per_node=2 trainer/train_pretrain.py \
  --data_path /mnt/boot/datasets/zzmind/pretrain_t2t.jsonl \
  --save_dir /mnt/boot/datasets/zzmind/out \
  --save_weight pretrain \
  --epochs 1 \
  --batch_size 8 \
  --max_seq_len 512 \
  --accumulation_steps 4 \
  --learning_rate 5e-4 \
  --warmup_steps 1000 \
  --total_tokens 65536000 \
  --log_interval 50 \
  --save_interval 500 \
  --num_workers 8 \
  2>&1 | tee /mnt/boot/datasets/zzmind/out/e1_$(date +%m%d_%H%M).log
echo "E1 done: $(date)"