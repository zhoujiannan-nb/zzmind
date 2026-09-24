# zzmind —— 从 0 训练一个有性格的小智能体

> 两个人（你 + AI 助手）作为研究伙伴，基于 [minimind](https://github.com/jingyaogong/minimind) 的训练链路，
> 在 2×RTX 4090 上从分词器一路走到一个 **0.5B、会聊天、会思考、会用工具、有自己性格** 的小智能体。
>
> 不追求榜单 SOTA，追求**每一步都亲手做过、都能讲清楚**。

---

## 🗺️ 学习路线图（一个过程一个文件夹）

| 阶段 | 文件夹 | 目标 | 状态 |
|---|---|---|---|
| 00 | [`00_env/`](00_env/) | 训练环境与数据侦察 | ✅ 完成 |
| 01 | [`01_tokenizer/`](01_tokenizer/) | 分词器（BPE / chat template） | ✅ 已完成（在另一台机器学习） |
| 02 | [`02_architecture/`](02_architecture/) | 模型架构：亲手看懂 0.5B 的每一个零件 | 🔵 **当前阶段** |
| 03 | [`03_pretrain/`](03_pretrain/) | 预训练：在 7.8G 中文语料上从零学语言 | ⬜ 未开始 |
| 04 | [`04_sft/`](04_sft/) | 指令微调：教会它"对话" | ⬜ 未开始 |
| 05 | [`05_alignment/`](05_alignment/) | 性格对齐：用 DPO 注入"人格" | ⬜ 未开始 |
| 06 | [`06_agent/`](06_agent/) | 智能体：工具调用 + Agentic RL | ⬜ 未开始 |
| 07 | [`07_inference/`](07_inference/) | 推理部署：Web Demo / OpenAI API | ⬜ 未开始 |
| 08 | [`08_zz_agent/`](08_zz_agent/) | 最终产品：我们的专属小智能体 | ⬜ 未开始 |

每个阶段文件夹里的 README 固定结构：
**目标 → 核心概念 → 代码地图 → 学习清单 → 实验任务 → 记录区（notes.md）→ 验收标准**。
你学完一个阶段，在 `notes.md` 里记录理解与踩坑，助手把下一阶段的文件夹写好，我们推进。

## 🎯 我们要训的模型（0.5B）

沿用 minimind 的 Qwen3 风格架构（RoPE + GQA + QK-Norm + SwiGLU + RMSNorm），把尺寸放大到 0.5B。
下表是**设计工作坊的默认值（均衡方案 B）**，最终规格以 `02_architecture/design/DESIGN.md` 的拍板结果为准：

| 超参数 | 值 | 备注 |
|---|---|---|
| hidden_size | 1280 | |
| num_hidden_layers | 24 | |
| 注意力头 / KV 头 | 16 / 8 (head_dim 80) | GQA，省 KV cache |
| intermediate_size | 4032 | SwiGLU，自动公式 ceil(h·π/64)·64 |
| vocab_size | 6400 | minimind 自带 BPE 词表（已在 01 阶段学习） |
| max_position_embeddings | 32768 | 可开 YaRN 长上下文 |
| tie_word_embeddings | true | 0.5B 规模下 embedding 与 lm_head 共享 |

参数量估算：**≈ 498M**（24 层 × ~20.4M + 8.2M embedding）。
2×4090（48G VRAM）bf16 + 梯度累积，seq 512~1024 预训练完全够用。

## 💡 0.5B 能做什么（我们的想象力边界）

0.5B 不是通用大模型，强求"什么都懂"会失望。但它做**定位清晰的小智能体**刚好够用：

1. **人格陪伴**：一个有固定性格、口癖、世界观的角色（SFT + DPO 塑造），聊天质量稳定、风格统一。
2. **2~4 个工具的调用**：时间/天气/计算/搜索/记事本——工具调用是"格式 + 少量参数"，小模型完全学得会。
3. **轻量推理**：数学应用题、简单逻辑题（用思考模式 `think` token 训练，类似 QwQ 思路）。
4. **个人助理**：读笔记、摘要、日程提醒——靠工具，不靠脑子。
5. **可控性**：参数少 → 训练快 → 可以反复试错换人格，这是 70B 大模型做不到的"性格实验自由"。

> 具体人格与工具集在 **05 阶段**定案，先留一个候选清单在 `08_zz_agent/` 里慢慢想。

## 🖥️ 环境

| 项 | 值 |
|---|---|
| 训练机 | `ssh node05`，2× NVIDIA RTX 4090（24G×2） |
| 软件 | Python 3.12.4 / torch 2.8.0+cu128 / transformers 4.57.6 |
| 数据 | `/mnt/boot/datasets/zzmind/`（见 `00_env/README.md` 明细） |
| 底座代码 | `D:\project_ai\minimind`（本地 Windows，ssh 到 node05 跑） |
| 仓库 | https://github.com/zhoujiannan-nb/zzmind |

## 🤝 协作流程

1. 助手写好**当前阶段**文件夹（概念 + 清单 + 实验脚本）。
2. 你按学习清单自学，在 `notes.md` 记理解/问题/踩坑。
3. 助手陪你跑实验（node05 上），把关键输出贴回 `notes.md`。
4. 达到**验收标准** → 主 README 状态列打勾 → 进入下一阶段。
5. 每阶段结束 `git push`，仓库就是学习档案本身。
