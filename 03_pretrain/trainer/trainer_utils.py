#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练工具函数集合（对齐 minimind 底座行为，逐个讲清"为什么"）。"""
import os
import sys
__package__ = "trainer"
# 让本目录 03_pretrain/ 的 dataset.py / model.py 优先于第三方同名包（如 pip 的 dataset）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer

from model import MiniMindConfig, MiniMindForCausalLM  # 我们自己的 0.5B 实现（03_pretrain/model.py）


def is_main_process():
    """多卡时只有 rank0 打印/存盘，避免 2 份重复输出互相刷屏。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def init_distributed_mode():
    """DDP 初始化：读环境变量 RANK/LOCAL_RANK（torchrun 会注入），
    nccl 通信 + 本卡 cuda 设备绑定。RANK 不存在 = 单卡模式，返回 0。"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式（本地 smoke / 单卡调试）
    dist.init_process_group(backend="nccl")          # 梯度 all-reduce 的通信后端
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed):
    """全链路定种：Python 随机 / numpy / torch / cuda，让实验可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(current_step, total_steps, lr, warmup_steps=0):
    """cosine 衰减 + warmup。
    - warmup：前 warmup_steps 步学习率从 0 线性升到峰值 —— 模型还一片混沌时
      大 lr 会让残差流爆掉，先小步走稳（llm 训练里"显而易见的常识"是踩坑踩出来的）。
    - cosine：过峰后平滑降到 0.1×lr，晚期小步细修，比线性衰减更稳。
    minimind 原版没有显式 warmup（cosine 起点自带缓冲），我们加上以对齐 README 拟定。"""
    def _warmup_factor(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, step / warmup_steps)

    cosine = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    return cosine * _warmup_factor(current_step)


def lm_checkpoint(lm_config, weight='pretrain', model=None, optimizer=None,
                  epoch=0, step=0, tokenizer_path=None, save_dir='../checkpoints', **kwargs):
    """断点保存/续训两块：
    - 权重 pth（fp16 半精度落盘，供 generate 用）；
    - resume pth（model+optimizer+step 全量，供崩溃后续训 —— optimizer 状态不存，
      续训 = 白练，所以必须带上）。"""
    os.makedirs(save_dir, exist_ok=True)
    base = f'{save_dir}/{weight}'
    ckp_path = f'{base}.pth'
    resume_path = f'{base}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 权重：fp16 足够（推理用），省一半磁盘
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        # resume：带 optimizer、epoch、step
        resume_tmp = resume_path + '.tmp'
        torch.save({'model': {k: v.half().cpu() for k, v in state_dict.items()},
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'step': step}, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp = torch.load(resume_path, map_location='cpu')
            # 卡数变化时把绝对步数按卡数折算（梯度累积语义不同）
            saved_ws = ckp.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp['step'] = ckp['step'] * saved_ws // current_ws
                Logger(f'GPU 数量变化({saved_ws}→{current_ws})，step 自动折算为 {ckp["step"]}')
            return ckp
        return None


def init_model(lm_config, from_weight='none', tokenizer_path=None, save_dir='../out', device='cuda'):
    """建模型 + 分词器；from_weight 非 none 时从上次权重/续训点续。"""
    if tokenizer_path is None:
        tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'model')  # 默认拿 01 阶段训好的 6400 词表
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}.pth'
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f'Model Params: {total:.2f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """把数据集按 batch_size 切成批；支持 skip 前 N 个 batch（续训时跳过已训部分）。"""
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler, self.batch_size, self.skip_batches = sampler, batch_size, skip_batches

    def __iter__(self):
        batch, skipped = [], 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if batch and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)