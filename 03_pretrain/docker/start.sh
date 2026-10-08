#!/bin/bash
# ============================================================
# zzmind 预训练 · 启动（宿主机 node05 侧）
#   容器: zzmind-pre-trainning (vllm/vllm-openai, GPU 0+1, 端口 7791)
#   数据: /mnt/boot/datasets/zzmind  ->  容器 /data   (宿主机地址不动)
#   进度: /home/ai-servers/zzmind/progress.json <-> 容器 /opt/zzmind
# 用法:
#   bash /home/ai-servers/zzmind/start.sh          # 自动: 读进度文件, 有断点则续训, 否则从头
#   bash /home/ai-servers/zzmind/start.sh fresh    # 强制从头训 (进度文件归档)
# 面板: http://<node05 IP>:7791/
# ============================================================
set -e
BASE=/home/ai-servers/zzmind
CONT=zzmind-pre-trainning
HOST_DATA=/mnt/boot/datasets/zzmind
PROG=$BASE/progress.json
IMAGE=vllm/vllm-openai:latest
PORT=7791
TOTAL=183105   # 1.5e9 token / 8192 token每步

# ---- 1. 容器没起就拉起（没有就创建） ----
if ! docker ps --format '{{.Names}}' | grep -q "^$CONT$"; then
  if docker ps -a --format '{{.Names}}' | grep -q "^$CONT$"; then
    echo "[start] 启动已有容器 $CONT"
    docker start $CONT
  else
    echo "[start] 创建容器 $CONT"
    docker run -d --name $CONT \
      --gpus '"device=0,1"' \
      --restart unless-stopped \
      -p $PORT:$PORT \
      --shm-size=32g \
      -v $HOST_DATA:/data \
      -v $BASE:/opt/zzmind \
      --entrypoint python3 \
      $IMAGE /opt/zzmind/monitor.py
  fi
  sleep 3
fi

# ---- 2. 防重复拉起 ----
if docker exec $CONT bash -c 'for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "train_pretrain[.]py" && exit 0; done; exit 1' 2>/dev/null; then
  echo "[start] 训练已在运行, 退出 (面板: http://127.0.0.1:$PORT/)"
  exit 0
fi

# ---- 3. GPU 占用提醒（教师模型在跑会抢显存） ----
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
echo "[start] GPU 显存占用: $USED MiB (若 >5000 说明有别的任务在用卡, 注意 OOM)"

# ---- 4. 读进度文件决定起点 ----
MODE=${1:-auto}
RESUME=""
if [ "$MODE" = "auto" ] && [ -f "$PROG" ] && [ -f "$HOST_DATA/out/pretrain_resume.pth" ]; then
  STEP=$(P="$PROG" python3 -c "import json,os;print(json.load(open(os.environ['P'])).get('step',0))" 2>/dev/null || echo 0)
  if [ "$STEP" -gt 0 ] && [ "$STEP" -lt "$TOTAL" ]; then
    RESUME="--from_resume 1"
    echo "[start] 读进度文件: step=$STEP/$TOTAL -> 续训"
  else
    echo "[start] 进度为空或已跑完 -> 从头训"
  fi
else
  if [ "$MODE" = "fresh" ] && [ -f "$PROG" ]; then
    mv "$PROG" "$PROG.$(date +%m%d_%H%M).bak"
  fi
  [ -z "$RESUME" ] && echo "[start] 从头训"
fi

# ---- 5. 拉起训练（容器内, 双卡 DDP） ----
TS=$(date +%m%d_%H%M)
docker exec -d $CONT bash -c "cd /data/03_pretrain && torchrun --nproc_per_node=2 \
  trainer/train_pretrain.py \
  --data_path /data/pretrain_t2t.jsonl,/data/router_for_pretrain.jsonl \
  --save_dir /data/out --save_weight pretrain --epochs 1 \
  --batch_size 8 --max_seq_len 512 --accumulation_steps 4 \
  --learning_rate 5e-4 --warmup_steps 1000 \
  --total_tokens 1.5e9 --log_interval 50 --save_interval 2000 --num_workers 8 \
  $RESUME 2>&1 | tee /data/out/pretrain_$TS.log"
echo "[start] 训练已拉起: 日志 /data/out/pretrain_$TS.log, 面板 http://127.0.0.1:$PORT/"
