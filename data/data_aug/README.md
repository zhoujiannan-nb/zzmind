# data_aug · 多意图路由语料合成（✅ 已完成）

> 本目录是 **数据准备阶段**（`data/`）的核心任务：用 27B 教师模型（Qwen3.8-27B-FP8）为 0.5B 小智能体
> **minicode** 预合成 68 万条"任务调度对话"语料，教会它把用户请求路由到合适的能力档模型卡片。
> 产出主要供 **03_pretrain**（路由语料），人设类供 **04_sft / 05_alignment**。
> 合成任务在 **node05** 上运行，**2026-10-04 01:58 全部完成**（689,742 条 > 目标 679,995），
> 此前通过 **txy:7788** 看板远程监控（http://124.223.88.17:7788/）。

## 双教师端点（v3，2026-09-28 上）

两个 27B 教师实例，**都走 ubasic 平台代理**（过平台计费/管理，不直连），每端点独立并发窗口、看板热调：

| 端点 | 平台代理地址 | 后端 | 默认并发 | 备注 |
|---|---|---|---|---|
| A · qwen3.8 | `http://ubasic.yuanshi-sec.com:5708/proxy/model/qwen3.8` | 原教师节点 | 4 | 白天留算力给 node05 其他任务 |
| B · qwen3.8-node05 | `http://ubasic.yuanshi-sec.com:5708/proxy/model/qwen3.8-node05` | node05 本机 SGLang（`/home/ai-servers/Qwen3.8-27B-FP8`，nginx 9007 → sglang 8081，v24 配置 TP2+DFLASH） | 10 | 09-28 上线；node05 重启后平台曾报"计算节点离线/502"，节点恢复后需平台侧重查转发 |

- 端点配置存 `manifest.db` kv 表 `endpoints`（JSON 数组：name/url/api_key/model/workers），
  看板改并发/地址 → `POST /api/control {key:"endpoint", idx, workers|url|api_key}` → **热生效不重启**
- 单端点并发范围 0~16（0 = 停用该端点，流量自动全走另一端）；总窗口 = 各端点之和
- 某端点挂了（502/离线）：该端点样本重试 3 次后丢弃重排，不影响另一端产出；
  平台错误体（`{"error":...}` 无 choices）也按失败重试
- 端点 api_key 在 `C:\Users\flycat666\.mini_code\config.json` 的 models 列表里可查（与平台模型一一对应）

## 三机架构

```
node05 (116.148.192.34:8002)                 txy (124.223.88.17)
┌─────────────────────────────────────┐       ┌────────────────────────────────────┐
│ /mnt/boot/datasets/zzmind/data_aug  │       │ /home/ubuntu/minicode_dashboard/   │
│  gen.py run  (PID 常驻, API :7799)  │◄──────│  ssh -N -p 8002 -L 7799:127.0.0.1: │
│  manifest.db  routing_*.jsonl       │ ssh   │            7799 node05 隧道        │
│  gen.log  model_pool.json           │ 隧道  │  txy_proxy.py :7788               │
└─────────────────────────────────────┘       │  = index.html(静态) + /api/* 代理  │
  教师 API: ubasic.yuanshi-sec.com:5708       └────────────────────────────────────┘
  (Qwen3.8-27B-FP8, SGLang)                        ▲
                                                    │ 浏览器
                                              http://124.223.88.17:7788/
```

- 教师模型走外部 API（`http://ubasic.yuanshi-sec.com:5708/proxy/model/qwen3.8/chat/completions`），
  **不占 node05 的 4090 显存**，node05 只是生成器 + 数据落地。
- txy 上的 ssh 隧道是前台/常驻 ssh 会话；隧道断了看板自动回退演示模式（数据变假，看 badge 判断）。

## 代码地图

| 文件 | 作用 |
|---|---|
| `gen.py` | 合成引擎（v3 双端点，当前在跑）：manifest 调度 + 双端点独立并发窗口 + 校验去重 + Web API(7799) |
| `gen.py.bak_v1` | v1 存档：无思考档位、`enable_thinking` 布尔开关、固定重试 |
| `gen.py.bak_v2_limit` | v2 存档：单端点 + 全局 workers/limit 开关（node05 上另有 bak_v3_keybug/bak_v3a 过渡版） |
| `model_pool.json` | 50 个教师模型卡片池（GPTQA 分 + 擅长领域），每条样本随机抽 2~5 张当"可用模型" |
| `start_gen.sh` | 启动脚本：rm STOP + nohup 拉起 |
| `dashboard/index.html` | 单文件看板（7 类型 × 15 领域格子矩阵 + KPI + 开关 + 双端点配置 + 日志 + 样本预览） |
| `dashboard/txy_proxy.py` | txy:7788 服务：静态目录 + `/api/*` 反代到 127.0.0.1:7799 |
| `dashboard/API.md` | 看板 API 契约（summary/cells/preview/log/control） |

## 数据设计（gen.py 里的核心常量）

- **7 类型**（占比）：simple 简单 18% / normal 正常临界 22% / complex 复杂 23% /
  self 自己能答 10% / fuzzy 模糊追问 12% / fallback 兜底不确定 10% / persona 人设 5%
- **15 领域**：代码 数学 中文写作 生物 日常 健康 历史地理 财经 搜索 文件管理 日程
  教育问答 游戏娱乐 图像媒体 科普
- **目标 680,000 条**（TOTAL），每格 = 680000×ratio/15，按 1000 条/批切分（可断点续跑）
- **人设**：minicode（毒舌/冷静/精准/自信/偶尔俏皮），制作人 zhoujiannan-nb
- **思考链**：80% 模板引导（"思考："行按 意图分析→难度判断→决策依据 写）+
  20% 教师真实思考（THINK_RATIO=0.2，SGLang `reasoning_effort` 分档 low/medium/xhigh，
  按样本类型匹配档位，见 `THINK_MAP`；实测 high/minimal/max → 400，只有 none/low/medium/xhigh 有效）
- **校验**：必须以 `系统：` 开头、含 `用户：/思考：`、按类型含 `路由：` 或 `回复：`、
  complex 必须有点名理由、全文 120~900 字；sha1 去重
- **输出**：persona 类单独落 `persona_posttrain.jsonl`（人设重复是零信息量，不进预训练，留给后训练），
  其余进 `routing_multi_intent.jsonl`
- **热开关**（sqlite kv 表，/api/control 或看板操作）：`running` 启停 / `fallback` 兜底类 /
  `think` 真实思考 / `endpoints` **双端点配置**（每端点 url/api_key/model/workers，0~16，热生效）
- **保护**：5 分钟无产出自动退出（API 挂了别空转；全端点并发=0 时是软暂停不触发）；
  touch `STOP` 文件优雅停止
- **node05 教师服务**：`/home/ai-servers/Qwen3.8-27B-FP8/`（start.sh 自动拉起 vllm_guard 防抢卡；
  健康端点 `/health`；chat 路由 `/v1/chat/completions`；9 并发槽 mamba48/5）

## 运行手册

```bash
# 生成器（node05，数据目录）
ssh node05
cd /mnt/boot/datasets/zzmind/data_aug
python3 gen.py run              # 断点续跑（manifest.db 记录进度）
python3 gen.py status           # 进度
python3 gen.py preview complex 代码 3   # 抽查样本
python3 gen.py smoke 60         # 冒烟（不占配额）
touch STOP && python3 gen.py run   # 优雅停止方式：touch STOP（跑着的进程 0.4s 轮询）

# 看板（txy）
ssh txy
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
    -p 8002 -L 7799:127.0.0.1:7799 root@116.148.192.34 &   # 隧道
cd /home/ubuntu/minicode_dashboard
python3 txy_proxy.py &        # :7788 → 127.0.0.1:7799
# 浏览器: http://124.223.88.17:7788/
```

注意：`gen.py` 顶部硬编码了教师 API 地址和 KEY（内网代理，密钥可暴露级别 = 内网）。
`BASE/KEY/MODEL` 换教师模型时改这三行。

## 当前状态（2026-10-08 15:30 复核，✅ 已完成）

- **语料生成完毕**：10-04 01:58 收尾（"全部批次完成！本次 544411 条，用时 5.0天15.0时"），gen.py 进程已退出
- 总进度：**689,742 / 679,995（101.4%）**，7 类型全部达标（批粒度超额），dup 0，fail 批次 0
- 分类型：simple 123,623/122,400 ｜ normal 151,826/149,595 ｜ **complex 158,820/156,405** ｜
  self 68,884/67,995 ｜ fuzzy 83,097/81,600 ｜ fallback 68,975/67,995 ｜ persona 34,517/34,005
- 数据文件：`routing_multi_intent.jsonl` **655,176 行 / 1.02GB（≈0.30B token）**，
  `persona_posttrain.jsonl` **34,566 行 / 52MB（≈0.015B token）**
- 10-08 抽查：sha1 零重复；~17% 条目标题前有教师输出多带的 `\n\n`（gen.py 校验 strip 后通过，外观问题不影响训练）
- **10-08 已重新 merge**：全量 65.5 万条并入 03（`/mnt/boot/datasets/zzmind/router_for_pretrain.jsonl`
  948MB / 0.305B token；旧 28.2 万条版备份为 `.bak_282k_0929`——那版是 09-29 语料 41% 时 merge 的）
- node05 的 `03_pretrain/` 代码与本地一致（token 预算公式修正 + start_pretrain.sh），随时可开训

### 重启/操作清单

```bash
ssh node05 "ps aux | grep 'gen.py run' | grep -v grep"          # 看进程
ssh node05 "bash /mnt/boot/datasets/zzmind/data_aug/start_gen.sh"   # 没跑就拉起（rm STOP + nohup）
# 改并发/地址：看板 http://124.223.88.17:7788/ 教师端点区，改完点"应用"即热生效
# 或命令行:
curl -X POST http://127.0.0.1:7799/api/control -d '{"key":"endpoint","idx":1,"workers":6}'
```

## 接手待办

- [x] 关注 simple 收尾后 normal/complex 的重复率（最终 fail 批次全 0，sha1 零重复）
- [x] complex 占比最大（156K）且用 xhigh 思考档，吞吐会掉，盯 ETA（已完成 158,820 条）
- [x] 语料落满后并入 03_pretrain 语料（10-08 全量 65.5 万条 merge 完成；persona 留 04_sft/05_alignment）
- [ ] **03 开训**：`bash /mnt/boot/datasets/zzmind/03_pretrain/start_pretrain.sh`（1.5e9 token ≈ 183,105 步 ≈ 15~30h，GPU 空闲）
- manifest.db / jsonl 属于数据产物，不进 git（在 node05 数据盘）
