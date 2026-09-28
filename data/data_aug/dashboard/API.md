# 进度看板 · API 契约

看板：`03_pretrain/progress/index.html`（单文件，无依赖）。
**演示模式**：直接双击打开，自带模拟数据（进度会自己爬）。
**实时模式**：部署在 `ssh txy` 的 **7788 端口**；生成器在 node05 起 HTTP API（端口 7799，CORS *），
看板每 5s 拉取；fetch 失败自动回退演示模式。跨机用 `?api=http://<node05>:7799/api` 指定。

## 大纲常量（与看板一致）
- 7 类型：simple 简单 18% / normal 正常临界 22% / complex 复杂 23% / self 自己能答 10% /
  fuzzy 模糊追问 12% / fallback 兜底不确定 10% / persona 人设 5%
- 15 领域：代码 数学 中文写作 生物 日常 健康 历史地理 财经 搜索 文件管理 日程 教育问答 游戏娱乐 图像媒体 科普
- 目标 680,000 条；每格 = target×ratio/15；批次 1000 条/批（可断点续跑）
- 思考档位：中 / 高（80% 模板引导 + 20% 真实思考，可经 /api/control 切换）
- 人设：minicode（智子风：毒舌/冷静/精准/自信/偶尔俏皮）· 制作人 zhoujiannan-nb，所有样本 system 带身份块

## GET /api/summary
```json
{
  "running": true,
  "fallback": true,
  "think": true,
  "generated": 152340,
  "target": 680000,
  "throughput_tps": 441.6,
  "tokens": 56300000,
  "dup": 1283,
  "failed_batches": 2,
  "started_at": "2026-09-25T20:00:00+08:00",
  "eta_seconds": 198000
}
```

## GET /api/cells
```json
[
  {"type": "simple",  "domain": "代码", "target": 9067, "done": 9000, "status": "done"},
  {"type": "normal",  "domain": "代码", "target": 11334,"done": 4500, "status": "run"},
  {"type": "complex", "domain": "生物", "target": 11334,"done": 0,    "status": "pend"},
  {"type": "self",    "domain": "日常", "target": 4533, "done": 4000, "status": "fail"},
  {"type": "fuzzy",   "domain": "文件管理","target": 5440,"done": 5440,"status": "done"},
  {"type": "fallback","domain": "代码", "target": 4533, "done": 1200, "status": "run"},
  {"type": "persona", "domain": "日常", "target": 2267, "done": 0,    "status": "pend"}
]
```
status ∈ `pend | run | done | fail`

## GET /api/preview?type=simple&domain=代码&n=5
```json
[{"text": "系统：你是任务调度智能体。……\n用户：……\n思考：……\n路由：……"}]
```
按文本里 `系统：/用户：/思考：/工具调用：/工具结果：/路由：/回复：` 前缀着色（看板已实现）。

## GET /api/log?n=50
```json
["[ok] 批次 simple/代码 #0007 完成 1000 条（18m42s，451 tok/s）", "…"]
```

## POST /api/control
```json
{"key": "fallback", "enabled": false}
```
key ∈ `running`（生成启停）/ `fallback`（兜底类是否继续生成）/ `think`（真实思考 20% 开关）。
对当前正在跑的批次不生效，影响后续批次。

## 大纲常量（与看板一致）
- 7 类型：simple 简单类 18% / normal 正常临界类 22% / complex 复杂类 23% / self 自己能答 10% /
  fuzzy 模糊追问 15%→12% / fallback 兜底不确定 10% / persona 人设 5%
- 15 领域：代码 数学 中文写作 生物 日常 健康 历史地理 财经 搜索 文件管理 日程 教育问答 游戏娱乐 图像媒体 科普
- 目标 680,000 条；每格 = target×ratio/15；批次 1000 条/批（可断点续跑）
