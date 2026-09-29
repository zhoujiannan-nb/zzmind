#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路由语料并入 03 预训练：data_aug 产出 → 标准 {"text": ...} jsonl（可直接训）。

设计依据（DATA.md / DESIGN 预算）：
- 路由语料 68 万条（7 类型×15 领域），text 字段天然就是预训练格式
  （系统：/用户：/思考：/工具调用：/工具结果：/回复： 完整对话），无需改写。
- persona 类（5%）归 04_sft/05_alignment，不进预训练 → 此处按 type 过滤。
- 输出与 pretrain_t2t.jsonl 同格式，train_pretrain.py 按逗号分隔多文件混训。

用法（node05）:
    python3 merge_router.py --src ../data_aug/routing_multi_intent.jsonl \
        --out ..//router_for_pretrain.jsonl
输出: 标准 jsonl + 行数/avg字符/估算token（复用 6400 词表实测比率 0.758 tok/char）。
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='routing_multi_intent.jsonl 源文件')
    ap.add_argument('--out', required=True, help='输出标准 jsonl')
    ap.add_argument('--ratio', type=float, default=1.0, help='抽样并入比例（默认全部）')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    total = keep = skip = chars = 0
    with open(args.src, 'r', encoding='utf-8') as fin, \
         open(args.out, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            obj = json.loads(line)
            if obj.get('type') == 'persona':          # persona 留给 04/05，不进预训练
                skip += 1
                continue
            chars += len(obj['text'])
            keep += 1
            if args.ratio < 1.0 and keep % 100 < args.ratio * 100:
                continue                              # 简易按比例抽样
            fout.write(json.dumps({'text': obj['text']}, ensure_ascii=False) + '\n')

    tok_per_char = 0.758                            # E4 实测：6400 词表 1 字符 ≈ 1.32 token
    kept = total - skip
    print(f'源 {total} 条 | 过滤 persona {skip} | 输出 {kept} 条')
    print(f'  平均每条约 {chars / max(kept, 1):.0f} 字符 ≈ 估算 {chars * tok_per_char / max(kept, 1):.0f} token/条')
    print(f'  估算总量 ≈ {chars * tok_per_char / 1e9:.3f} B token')


if __name__ == '__main__':
    main()