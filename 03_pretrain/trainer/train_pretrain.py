#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zzmind-0.5B 预训练主入口（03 阶段）。

把你之前那句大白话翻成代码：
    数据准备好 → 输入 [B, N] token → 模型吐 logits → 算 CE loss → torch 自动倒走
    （autograd 记图 → backward 把梯度传回每个参数）→ AdamW 更新。
双卡：DDP 每卡一份模型、各吃一半 batch，反向完 all-reduce 梯度求平均再各自更新。

用法：
    # 单卡调试（本机 CPU/GPU）
    python 03_pretrain/trainer/train_pretrain.py --data_path <小文件> --epochs 1
    # 双卡正式（node05，2×4090）
    torchrun --nproc_per_node=2 03_pretrain/trainer/train_pretrain.py \
        --data_path /mnt/boot/datasets/zzmind/pretrain_t2t.jsonl
"""
import os
import sys
__package__ = "trainer"
# 让本目录 03_pretrain/ 的 dataset.py / model.py 优先于第三方同名包（如 pip 的 dataset）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model import MiniMindConfig
from dataset import PretrainDataset
from trainer.trainer_utils import (get_lr, Logger, is_main_process, lm_checkpoint,
                                   init_distributed_mode, setup_seed, init_model, SkipBatchSampler)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        if step > iters:
            break   # 达到 token 预算总步数即停（epochs=1 时只吃 ~1/3 语料，合 DESIGN 预算）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # 1. 当前步学习率（warmup + cosine）
        lr = get_lr(step, args.total_steps, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2. 前向：输入 [B,N] → logits → loss（bf16 autocast 全程）
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / args.accumulation_steps   # 累积时先除 N，等效大 batch

        # 3. 反向：torch 自动倒走计算图，梯度进每个参数 .grad
        scaler.scale(loss).backward()

        # 4. 攒满 accumulation_steps 个微批才真正更新一次
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 防 outlier 炸权重
            scaler.step(optimizer)          # AdamW 按梯度更新权重
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 5. 日志：loss / 吞吐 / ETA
        if step % args.log_interval == 0 or step == iters:
            spend = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            tps = args.batch_size * args.max_seq_len * args.accumulation_steps * args.gpus / max(spend / (step - start_step), 1e-9)
            eta_min = spend / max(step - start_step, 1) * (iters - step) / 60
            Logger(f'Epoch:{epoch + 1}/{args.epochs} step:{step}/{iters} loss:{cur_loss:.4f} '
                   f'lr:{lr:.2e} {tps / 1e3:.1f}k tok/s ETA:{eta_min:.0f}min')

        # 6. 周期存档（只主进程做；权重 + resume 双份）
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                          optimizer=optimizer, epoch=epoch, step=step, save_dir=args.save_dir)
            model.train()

        del input_ids, labels, res, loss

    # 尾部微批收口（步数不是 accum 整数倍时最后一点梯度没 step）
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='zzmind-0.5B Pretraining')
    parser.add_argument('--save_dir', type=str, default='../out', help='权重/续训点保存目录')
    parser.add_argument('--save_weight', default='pretrain', type=str, help='权重前缀名')
    parser.add_argument('--data_path', type=str, required=True, help='预训练 jsonl（每行 {"text": ...}）')
    parser.add_argument('--tokenizer_path', default=None, type=str, help='01 阶段 6400 词表目录（默认 ../model）')
    parser.add_argument('--epochs', type=int, default=1, help='数据过几遍（1.5B token ≈ 1/3 语料，一遍即可）')
    parser.add_argument('--batch_size', type=int, default=8, help='每卡微批大小（显存实测后可调）')
    parser.add_argument('--max_seq_len', type=int, default=512, help='训练序列长（README 拟定：512 起步）')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='峰值学习率')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='warmup 步数（比例 ≈ 2%% 总步）')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积（显存补足 batch）')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--total_tokens', type=float, default=1.5e9, help='token 预算 → 折算总步数')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 读数据线程数')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=2000, help='每多少步存档')
    parser.add_argument('--from_weight', default='none', type=str, help='从哪个权重续（none=从头）')
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help='1=读 resume.pth 自动续训')
    parser.add_argument('--use_compile', default=0, type=int, choices=[0, 1], help='torch.compile 加速（可选）')
    args = parser.parse_args()

    # ===== 1. 环境：DDP / 随机种子 =====
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    args.gpus = dist.get_world_size() if dist.is_initialized() else 1

    # ===== 2. 模型配置（定稿默认即 zzmind-0.5B，无需改） =====
    lm_config = MiniMindConfig()
    os.makedirs(args.save_dir, exist_ok=True)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume == 1 else None

    # ===== 3. 混合精度上下文 =====
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if args.device == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    # ===== 4. 模型 / 数据 / 优化器 =====
    model, tokenizer = init_model(lm_config, args.from_weight, args.tokenizer_path, args.save_dir, args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))   # bf16 不需要缩放
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ===== 5. 续训恢复状态 =====
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
        Logger(f'[resume] 从 epoch {start_epoch} step {start_step} 续训')

    # ===== 6. 编译 / DDP 包装 =====
    if args.use_compile == 1:
        model = torch.compile(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ===== 7. 总步数：token 预算 / 每步有效 token =====
    per_step_tokens = args.batch_size * args.max_seq_len * args.accumulation_steps * args.gpus
    args.total_steps = int(args.total_tokens / per_step_tokens)
    Logger(f'有效 batch：{args.batch_size}×{args.accumulation_steps}×{args.gpus}卡 '
           f'= {per_step_tokens} token/step；预算 {args.total_tokens / 1e9}B token → {args.total_steps} 步')

    # ===== 8. 开训 =====
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)   # 每 epoch 换 shuffle，避免多卡看到同一批
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(sampler or torch.randperm(len(train_ds)).tolist(),
                                         args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers,
                            pin_memory=('cuda' in args.device))   # CPU 模式别 pin，否则会占 cuda:0
        train_epoch(epoch, loader, args.total_steps, start_step)

    # ===== 9. 收尾 =====
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()