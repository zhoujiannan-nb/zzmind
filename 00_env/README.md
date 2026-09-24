# 00 · 环境与数据侦察 ✅

## 目标
确认"在哪训、用什么训、训什么"——把环境摸清楚，后面所有阶段的地基。

## 已完成事项
- [x] `ssh node05` 连通
- [x] GPU 确认：2× NVIDIA RTX 4090（24G 显存 ×2）
- [x] 软件栈：Python 3.12.4 / torch 2.8.0+cu128（CUDA 可用）
- [x] 数据落盘确认（用户给的 `dataset` 是 `datasets` 的笔误）：

| 文件 | 大小 | 行数 | 格式 |
|---|---|---|---|
| `pretrain_t2t.jsonl` | 7.8G | 8,468,827 | `{"text": "..."}` 纯文本，中文为主 |
| `sft_t2t_mini.jsonl` | 1.7G | 905,718 | `{"conversations": [{role, content, reasoning_content?}, ...]}` |
| `dl_pretrain.log` / `dl_sft.log` | 13M / 2.6M | - | 下载日志（数据来源线索，后面 03 阶段再看） |

## 关键结论
- 预训练语料约 **847 万条**，SFT 约 **90 万轮对话**，量级足够 0.5B 从 0 训起。
- SFT 数据带 `reasoning_content` 字段 → 天然支持"思考模式"训练（Qwen3 风格），04/06 阶段要用。
- 2×4090 足够 0.5B 全参数预训练（bf16），无需 MoE 也能跑。

## 下一步
进入 02_architecture：先不动 GPU，把模型结构彻底看懂。
