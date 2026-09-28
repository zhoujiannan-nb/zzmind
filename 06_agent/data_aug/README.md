# data_aug · 多意图路由语料合成（node05 正在跑的任务）

> 本目录是 **06_agent 阶段的数据前置**：用 27B 教师模型（Qwen3.8-27B-FP8）为 0.5B 小智能体
> **minicode** 预合成 68 万条"任务调度对话"语料，教会它把用户请求路由到合适的能力档模型卡片。
> 合成任务当前在 **node05** 上持续运行（`/mnt/boot/datasets/zzmind/data_aug/gen.py`），
> 通过 **txy:7788** 看板远程监控（http://124.223.88.17:7788/）。

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
| `gen.py` | 合成引擎（v2，当前在跑）：manifest 调度 + 并发窗口 + 校验去重 + Web API(7799) |
| `gen.py.bak_v1` | v1 存档：无思考档位、`enable_thinking` 布尔开关、固定重试 2/4s |
| `model_pool.json` | 50 个教师模型卡片池（GPTQA 分 + 擅长领域），每条样本随机抽 2~5 张当"可用模型" |
| `dashboard/index.html` | 单文件看板（7 类型 × 15 领域格子矩阵 + KPI + 开关 + 日志 + 样本预览） |
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
  `think` 真实思考 / `workers` 并发（2~12，热更新滑动窗口）/ `limit` **限流开关**（白天留算力，
  开启时并发钳到 `LIMIT_WORKERS=4`，不动 workers 值，关闭立即恢复）
- **启动脚本**：`start_gen.sh`（rm STOP + nohup 拉起，2026-09-28 node05 失联后重启用它）
- **保护**：5 分钟无产出自动退出（API 挂了别空转）；touch `STOP` 文件优雅停止

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

## 当前状态（2026-09-28 接手时）

- ⚠️ **node05 当日 09:24 起失联**（k8s 节点 zone76-node05 NotReady，09:18 最后心跳；
  机器 ping 得通但 sshd/kubelet 用户态服务卡死，疑似宿主机 I/O 或资源问题，待运维/宿主机控制台处理）
- 失联前：旧进程 09:22:51 优雅退出（in-flight 全收齐，数据一致）；新进程未及启动
- 失联前进度：145,064 / 680,000（21.3%）
  - simple 118,831/122,400（收尾，5 批 failed——重复率太高没填满，可接受）
  - normal 26,233/149,595（进行中）
  - complex/fuzzy/fallback/self/persona 未开始
- 吞吐：约 700~1100 tok/s（并发 8，看板可 +2 到 10）
- 数据文件：`routing_multi_intent.jsonl` ~423MB / 14.5 万行，`persona_posttrain.jsonl` 49 行

### node05 恢复后重启清单

```bash
ssh node05 "ps aux | grep 'gen.py run' | grep -v grep"      # 1. 看进程是否还活着（I/O 卡死时可能假活）
ssh node05 "bash /mnt/boot/datasets/zzmind/data_aug/start_gen.sh"   # 2. 没跑就拉起来
# 3. 看板 http://124.223.88.17:7788/ 勾"限流（并发→4）"→ 白天留算力；跑完别的事再取消
```

## 接手待办

- [ ] 关注 simple 收尾后 normal/complex 的重复率（failed 批次 = 4 次重试都撞 sha1 或校验不过）
- [ ] complex 占比最大（156K）且用 xhigh 思考档，吞吐会掉，盯 ETA
- [ ] 语料落满后：写 tokenize 脚本并入 03_pretrain 语料（persona 留 04_sft/05_alignment）
- [ ] manifest.db / jsonl 属于数据产物，不进 git（在 node05 数据盘）
