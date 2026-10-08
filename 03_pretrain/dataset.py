#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预训练数据集：7.8G jsonl 的流式随机访问（不把全量读进内存）。

与 minimind 的 PretrainDataset 行为对齐（bos + 正文 + eos + pad → labels 屏蔽 pad），
但底层换成「行偏移表」：内存只存每行的 (offset, length)（8.4M 行 ≈ 刚 200MB），
__getitem__ 时才 seek 读那一行 —— 7.8G 语料开机即用，不占内存不卡 load。
"""
import json
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """data_path 支持逗号分隔多文件（如 主语料,路由语料），按文件序拼接成一个数据集。"""
        super().__init__()
        self.tokenizer, self.max_length = tokenizer, max_length
        self._path = data_path   # 兼容单文件
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 每个文件一份偏移表 + 各自的基址，__getitem__ 全局下标 → (文件i, 文件内行号)
        # 二进制模式扫：tell()/seek() 是真字节偏移且快（文本模式 tell() 在部分 Python
        # 版本上慢 ~56 倍，847 万行要 12 分钟 → 曾把训练启动拖死，node05 容器实测）
        self.file_offsets, self.file_sizes = [], []
        self.file_cumsum = [0]   # 各文件行数累加，把全局下标映射到文件
        for path in data_path.split(','):
            offsets, sizes = [], []
            with open(path, 'rb') as f:
                while True:
                    off = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets.append(off)
                    sizes.append(len(line))
            self.file_offsets.append((path, offsets, sizes))
            self.file_cumsum.append(self.file_cumsum[-1] + len(offsets))

    def __len__(self):
        return self.file_cumsum[-1]

    def __getitem__(self, index):
        # 定位文件：cumsum[i] <= index < cumsum[i+1]
        fi = 0
        for i in range(1, len(self.file_cumsum)):
            if index < self.file_cumsum[i]:
                fi = i - 1
                break
        rel = index - self.file_cumsum[fi]
        path, offsets, _ = self.file_offsets[fi]
        with open(path, 'rb') as f:
            f.seek(offsets[rel])
            raw = f.readline()          # 二进制 tell/seek 是字节偏移，精确且快
        text = json.loads(raw.decode('utf-8', 'ignore'))['text']

        tokens = self.tokenizer(text, add_special_tokens=False,
                                truncation=True, max_length=self.max_length - 2).input_ids
        ids = [self.bos_id] + tokens + [self.eos_id]
        seq = ids + [self.pad_id] * (self.max_length - len(ids))    # 右补齐到定长
        input_ids = torch.tensor(seq, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100                      # pad 位不算 loss
        return input_ids, labels