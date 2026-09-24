# 01 · 分词器（Tokenizer）【已完成】

## 目标
把自然语言变成模型能吃的 token 序列，并学会"对话格式"。
（本阶段在另一台机器上学习完成，此处为结论归档。）

## 核心概念回顾
1. BPE 算法：从字符表出发，反复合并最高频的相邻 token 对，扩展词表。
2. 我们的词表：6400 个 token（比 Qwen 的 15 万小很多），专为小型模型设计，
   换来的是：embedding 参数少、训练快；代价是：每个汉字平均占更多 token。
3. 特殊 token：
   - 对话标记：system / user / assistant
   - 工具调用：tool_call 开闭、tool_response 开闭
   - 思考标记：think / think 结束（Qwen3 风格"思考模式"）
   - buffer 1~9：RL 阶段用的占位缓冲
4. chat_template：Jinja2 模板，把 messages 列表渲染成模型输入文本。
   工具调用时，system 里会注入 tools 的 JSON 签名说明。

## 关键文件
- minimind/model/tokenizer.json：词表本体
- minimind/model/tokenizer_config.json：特殊 token + chat_template
- minimind/trainer/train_tokenizer.py：分词器训练脚本（BPE 训练入口）

## 学习清单（回顾）
- [x] BPE 的合并过程能手推一遍
- [x] 为什么小词表对 0.5B 模型是合理的
- [x] chat_template 里 tool_call 和 think 标记的渲染逻辑
- [ ] 遗留问题：6400 词表下中文的 token/字符 比率实测（03 阶段顺手做）

## 下一步
进入 02_architecture：看懂 0.5B 模型的每一个零件。
