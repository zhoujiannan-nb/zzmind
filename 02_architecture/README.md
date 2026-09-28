# 02 · 模型架构：亲手设计 0.5B 的每一个零件

## 目标
从"0.5B、2×4090"出发，把 minicode 的完整架构亲手设计出来：
组件怎么选、每个维度分多少、最后落成 config 并用代码验证。
学完后要能讲清 config 里**每个数字**的设计决策，而不是"抄的 Qwen3"。

## 怎么做（四层，一层一层往下做）
设计本质是**带约束的求解**：先写约束（L0）→ 再选组件（L1）→ 再分预算（L2）→ 最后验证（L3）。
前两步不写代码，写的是"决定 + 理由"；代码放最后，让设计可被检查。

| 层 | 内容 | 产出 | 状态 |
|---|---|---|---|
| L0 | 约束确认：为什么 0.5B、它翻译过来是什么 | `design/L0_constraints.md` | ✅ |
| L1 | 骨架选择：注意力 / 位置编码 / 归一化 / FFN / embedding 是否共享 | `design/L1_skeleton.md` | ✅（6/6：GQA / RoPE / RMSNorm-pre / SwiGLU / tie / 无 dropout） |
| L2 | 容量分配：500M 怎么分给 hidden / 层数 / 头数 / 中间维度 | `design/L2_capacity.md` | ✅（定稿：1280×24×4032，497.8M） |
| L3 | 落地验证：config → 实例化 → 参数审计 → forward | `design/config_*.json` + `experiments/` | ✅（E1/E2 本地通过；E3 学习用脚本已给） |

> 主 README 里有一张"均衡方案 B"表（上一轮工作坊的成品）。
> 这次我们一层层重新推，推完和它 diff——推得出来 = 真懂了。

## 学习清单
随层填充，见各 `L*.md`。

## 实验任务（L3 用）
- **E1 forward walk**：实例化模型跑一次 forward，核对 shape 流和初始 loss
- **E2 param audit**：每个组件参数量拆分，验证总和 ≈ 0.5B
- **E3 mini GPT**：从 0 写一个最小 decoder（只为理解，生产不用它）

## 验收标准
1. config 通过 E2 审计：总参数与 0.5B 目标偏差 < 2%
2. 对 config 每个超参能讲清"为什么是这个值"
3. E1 forward：shape 正确，初始 loss ≈ ln(6400) ≈ 8.76
4. 通过后：主 README 状态改 ✅，进入 03_pretrain

## 下一步
03_pretrain：设计拍板后，用这份 config 开训。
