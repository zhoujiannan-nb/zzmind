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
        super().__init__()
        self.tokenizer, self.max_length = tokenizer, max_length
        self._path = data_path   # __getitem__ 时按行号 seek 回去读
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 一次扫描建偏移表：记录每行的起始字节 + 字节长（不读内容）
        self.offsets, self.sizes = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(off)
                self.sizes.append(len(line.encode('utf-8')))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self._path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(self.offsets[index])
            line = f.readline()          # 文本模式下 read(n) 按字符数算，会越界，必须 readline
        text = json.loads(line)['text']

        tokens = self.tokenizer(text, add_special_tokens=False,
                                truncation=True, max_length=self.max_length - 2).input_ids
        ids = [self.bos_id] + tokens + [self.eos_id]
        seq = ids + [self.pad_id] * (self.max_length - len(ids))    # 右补齐到定长
        input_ids = torch.tensor(seq, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100                      # pad 位不算 loss
        return input_ids, labels