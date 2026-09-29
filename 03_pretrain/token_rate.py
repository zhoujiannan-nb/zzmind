#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4: 实测 6400 词表的 token/字符 比率 + 全量 token 估算（01 遗留 / 03 预算依据）。

用法（node05）:
    python3 token_rate.py --jsonl ../pretrain_t2t.jsonl --tokenizer model --sample 20000
输出:
    抽样样本的 行数/字符数/token数 → token/字符 比率 → 按全量行数估算总 token
"""
import argparse
import json
import sys

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='待统计的 jsonl（每行 {"text": ...}）')
    ap.add_argument('--tokenizer', default='model', help='6400 词表目录')
    ap.add_argument('--sample', type=int, default=20000, help='抽样前 N 行')
    ap.add_argument('--total_rows', type=int, default=None, help='全量行数（不传则不估算全量）')
    ap.add_argument('--verbose', action='store_true', help='打印每行明细')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    rows = chars = tokens = 0
    dist = {}  # 每行 token 数分布（观察截断选 512 是否够）
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.sample:
                break
            text = json.loads(line)['text']
            n_char = len(text)
            n_tok = len(tok(text, add_special_tokens=False).input_ids)
            rows, chars, tokens = rows + 1, chars + n_char, tokens + n_tok
            bucket = (n_tok // 128) * 128
            dist[bucket] = dist.get(bucket, 0) + 1
            if args.verbose and i % 5000 == 0:
                print(f'  [{i}] 行={n_char}字符 ... token={n_tok}')

    print(f'抽样 {rows} 行: 字符合计 {chars:,}, token 合计 {tokens:,}')
    print(f'token/字符 = {tokens / chars:.4f}  (即 1 字符 ≈ {chars / tokens:.2f} token)')
    print(f'平均每行: {chars / rows:.0f} 字符 / {tokens / rows:.1f} token')
    print('每行 token 分布（128 桶）:')
    for b in sorted(dist):
        print(f'  ≤{b + 127:>5} token: {dist[b]:>6} 行 ({dist[b] / rows:5.1%})')

    print('---- 全量估算 ----')
    total_rows = args.total_rows
    if total_rows:
        est_tokens = total_rows * tokens / rows
        print(f'全量 {total_rows} 行 ≈ {est_tokens / 1e9:.2f} B token')
        print(f'   其中前 1.5B token ≈ 占全量 {1.5e9 / est_tokens * 100:.1f}%')
    else:
        print('（未传 --total_rows，跳过全量估算）')


if __name__ == '__main__':
    main()