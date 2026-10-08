#!/bin/bash
# ============================================================
# zzmind 预训练 · 停止（宿主机 node05 侧）
#   默认: 只停训练进程, 容器常驻 (7791 面板仍可用)
#   all : 连同容器一起停 (docker start zzmind-pre-trainning 可恢复)
# 用法:
#   bash /home/ai-servers/zzmind/stop.sh
#   bash /home/ai-servers/zzmind/stop.sh all
# 说明: 停止不丢进度 —— 每 2000 步自动存档, 下次 start.sh 读进度文件自动续训
# ============================================================
set -e
CONT=zzmind-pre-trainning

if ! docker ps --format '{{.Names}}' | grep -q "^$CONT$"; then
  echo "[stop] 容器 $CONT 没在运行"; exit 0
fi

find_pids() {
  docker exec $CONT bash -c \
    'for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "train_pretrain[.]py" && basename $p; done' 2>/dev/null
}

PIDS=$(find_pids | tr '\n' ' ')
if [ -n "$PIDS" ]; then
  echo "[stop] 停训练进程: $PIDS"
  docker exec $CONT bash -c "kill $PIDS 2>/dev/null || true"
  for i in $(seq 10); do
    [ -z "$(find_pids)" ] && break
    sleep 1
  done
  if [ -n "$(find_pids)" ]; then
    echo "[stop] 10s 未退出, 强杀"
    PIDS=$(find_pids | tr '\n' ' ')
    docker exec $CONT bash -c "kill -9 $PIDS 2>/dev/null || true"
  fi
  echo "[stop] 训练已停止 (最近一次存档 = 下次 start.sh 的续训点)"
else
  echo "[stop] 没有训练进程"
fi

if [ "$1" = "all" ]; then
  docker stop $CONT
  echo "[stop] 容器已停 (恢复: docker start $CONT)"
fi
