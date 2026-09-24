# 07 · 推理部署：让全世界（至少你自己）用上它

## 目标
把训好的 0.5B 模型变成可交互的服务：
Web 聊天页 + OpenAI 兼容 API + 推理性能摸底。

## 核心概念
1. **推理与训练的区别**：推理只做前向 + 自回归循环（02 的 generate），
   每生成 1 个 token 全模型跑一遍 → 速度瓶颈在显存带宽，不在算力。
2. **KV cache**：已生成位置的 K/V 缓存起来，新 token 只算增量。
   0.5B + GQA：seq 2048 时 KV cache ≈ 2 × 8头 × 80维 × 24层 × 2048 × 2字节
   ≈ 150MB/序列（GQA 让 KV 头减半，这就是 02 阶段说它"省显存"的地方）。
3. **OpenAI 兼容 API**：实现 /v1/chat/completions（streaming SSE），
   任何支持 OpenAI SDK 的客户端（Chatbox、各种 bot 框架）都能直接接。
4. **量化（可选）**：0.5B bf16 ≈ 1GB，CPU 都能跑；
   要上手机/浏览器再考虑 INT8/INT4（先不做，0.5B 没压力）。

## minimind 现成工具
- scripts/web_demo.py：Streamlit 聊天页
- scripts/serve_openai_api.py：OpenAI 兼容服务（FastAPI）
- scripts/chat_api.py：本地命令行聊天
- scripts/convert_model.py：权重格式转换

## 学习清单
- [ ] 手算 0.5B 的 KV cache 大小，与 02 阶段 GQA 的结论对账
- [ ] 为什么推理瓶颈是带宽不是算力（算术强度）
- [ ] streaming（流式输出）在 HTTP 层怎么实现的
- [ ] OpenAI API 的 messages/tools 字段怎么映射到我们的 chat_template
- [ ] 单卡 4090 上 0.5B 的 decode 速度理论上限怎么估

## 实验任务
- **E1 速度摸底**：测 prefill 速度（tokens/s）和 decode 速度（tokens/s），
  不同 batch size，记表。
- **E2 Web Demo**：web_demo.py 跑起来，接入 06 的工具循环，
  本地 + ngrok 内网穿透手机上聊。
- **E3 OpenAI API**：serve_openai_api.py 起服务，
  用 OpenAI SDK（base_url 指过来）发请求 + 测 streaming。
- **E4 压测（选做）**：并发 4/8/16，看吞吐曲线。

## 验收标准
1. 手机浏览器能通过 web demo 流畅对话（截图存档）
2. OpenAI SDK 三行代码调通（代码贴 notes）
3. 有性能数据表（prefill/decode/并发）
4. 通过后：主 README 状态改 ✅，进入 08_zz_agent

## 下一步
08_zz_agent：全部零件齐了——组装我们的专属小智能体。
