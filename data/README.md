# data · 数据准备（自合成语料）

## 目标
给各训练阶段准备**自有的**训练数据并管理到底。
现成数据（00 阶段侦察的 7.8G 预训练 + 1.7G SFT）解决"会说话"，
本目录的自合成数据解决"minicode 特有的能力"（当前 = 任务路由），
以及后续各阶段缺什么补什么。

## 目录
| 目录 | 内容 | 去向 | 状态 |
|---|---|---|---|
| [`data_aug/`](data_aug/) | 多意图路由语料合成：27B 教师蒸馏 68 万条"任务调度对话"（7 类型 × 15 领域） | `routing_multi_intent.jsonl`（~65 万）→ tokenize 后并入 **03_pretrain** 语料；`persona_posttrain.jsonl`（~3.4 万人设）→ **04_sft / 05_alignment** 后训练 | 🔵 进行中（node05 持续合成，进度/操作见该目录 README） |

## 数据流
```
data/data_aug/ ── routing_multi_intent.jsonl ──> tokenize ──> 03_pretrain 语料
             └── persona_posttrain.jsonl ────────────────────> 04_sft / 05_alignment
现成数据      ── pretrain_t2t.jsonl (7.8G) ─────────────────> 03_pretrain
             └── sft_t2t_mini.jsonl (1.7G) ─────────────────> 04_sft
```

## 约定
- 数据产物（jsonl / manifest.db / log）不进 git，只存 node05 数据盘
  （`/mnt/boot/datasets/zzmind/`），git 里只放生成器代码 + 文档。
- 每新增一类数据，加一个子目录 + 本表一行，README 写清：来源、规模、格式、去向。
