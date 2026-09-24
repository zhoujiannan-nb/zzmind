# 06 · 智能体：工具调用 + Agentic RL

## 目标
让有性格的小模型"长出手"：
1. 学会按格式输出 tool_call（01 阶段学的标记在这里用）
2. 学会多轮"思考 → 调工具 → 看结果 → 继续想"的循环
3. 用 RL（GRPO/PPO）强化：任务完成得好 → 奖励，格式错 → 惩罚

## 核心概念
1. **工具调用 = 格式学习**：system 里注入 tools 的 JSON 签名
   （01 阶段 chat_template 里的 # Tools 段），模型输出
   tool_call{"name":..., "arguments":...}。
   小模型学格式很快，难的是"什么时候调、参数填对"。
2. **Agent 循环**：prompt → 模型（思考 + 可能调工具）→ 工具执行 →
   结果以 tool_response 塞回 → 模型继续 → 直到给出最终答案。
   04 数据里的 tool/tool_response 角色就是为这个循环准备的。
3. **Agentic RL**：任务型奖励（答案对不对 / 任务完成没）直接当 reward，
   GRPO 不需要 value network（组内相对奖励），0.5B 规模更稳。
   minimind 的 train_grpo.py + rollout_engine.py 就是这套。
4. **工具集设计原则（0.5B 版）**：少而精。
   每个工具：名字动词化、参数 ≤ 3 个、描述 ≤ 2 句。
   候选：get_time / calculator / get_weather / search_web / note_add / note_list

## 工具调用训练路线
1. **阶段 A（SFT 为主）**：工具调用数据混入 SFT
   （minimind 的 SFT 数据已含 tool_call 样本），先能"按格式调对"
2. **阶段 B（RL 强化）**：设计任务环境（数学题要 calculator、
   日程问题要 get_time + note），完成度当 reward 跑 GRPO
3. **阶段 C（评估）**：eval_toolcall.py 跑工具调用准确率

## 配置（拟定）
| 项 | 值 |
|---|---|
| 底座 | 05 阶段 DPO 权重（性格 + 格式） |
| RL 算法 | GRPO（组大小 8） |
| 任务集 | ~200 个任务（3~5 个工具组合） |
| seq_len | 2048（多轮循环要长） |
| 显存 | 2×4090，rollout 用 GPU0 推理 + GPU1 训练（或轮转） |
| 时长 | 视收敛，预计 3~8 小时 |

## 学习清单
- [ ] 一次完整 agent 循环里，哪些 token 是模型生成的、哪些是环境塞的
- [ ] tool_call 格式错了（JSON 断行）会怎样，怎么在 RL 里惩罚
- [ ] GRPO 和 PPO 的区别（为什么没有 critic）
- [ ] reward 设计陷阱：reward hacking（模型钻空子拿分）举例
- [ ] 为什么 0.5B 工具数量要克制（给个定量直觉）
- [ ] 多轮循环里 KV cache 怎么复用（推理效率）

## 实验任务
- **E1 工具框架**：写本地工具执行器（Python dict 映射 + 超时保护），
  先接 calculator + get_time 两个最简单的。
- **E2 格式 SFT**：用工具调用样本再 SFT 一小段（1~2 小时），
  目标：tool_call 格式正确率 > 90%。
- **E3 GRPO 强化**：200 任务环境跑 GRPO，看任务成功率曲线。
- **E4 端到端演示**：接一个命令行聊天框，让它连续完成
  "现在几点了？帮我记个明天 9 点的会，顺便算下 23*47"。

## 验收标准
1. 单工具调用格式正确率 > 90%（eval_toolcall 输出贴 notes）
2. 多步任务成功率比 RL 前提升 ≥ 15 个百分点（贴对比）
3. E4 端到端演示录屏/日志存档
4. 通过后：主 README 状态改 ✅，进入 07_inference

## 下一步
07_inference：把它装进网页和 API，让人能"摸到"它。
