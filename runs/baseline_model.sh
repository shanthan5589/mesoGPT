#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere with:
# bash runs/baseline_model.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/experiments/baseline_model/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

RUN_TIMESTAMP="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
RUN_LOG="$LOG_DIR/baseline_${RUN_TIMESTAMP}.log"
GPU_LOG="$LOG_DIR/baseline_gpu_${RUN_TIMESTAMP}.csv"

exec > >(tee "$RUN_LOG") 2>&1

eval "$(conda shell.bash hook)"
conda activate mesogpt

echo "===== RUN INFORMATION ====="
echo "UTC start: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(hostname)"
python --version
python -c 'import torch; print(f"PyTorch: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA version: {torch.version.cuda}")'

echo "===== CPU INFORMATION ====="
lscpu

echo "===== MEMORY INFORMATION ====="
free -h

GPU_MONITOR_PID=""

stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    GPU_MONITOR_PID=""
}

trap stop_gpu_monitor EXIT INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "===== GPU INFORMATION ====="
    nvidia-smi
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv \
        --loop=5 > "$GPU_LOG" &
    GPU_MONITOR_PID=$!
else
    echo "nvidia-smi not found; GPU monitoring is unavailable."
fi

echo "===== BASELINE TRAINING ====="
/usr/bin/time -v python experiments/baseline_model/base_train.py \
    --context_length 1024 \
    --n_embed 768 \
    --n_layers 12 \
    --n_heads 12 \
    --dropout 0.1 \
    --batch_size 512 \
    --learning_rate 3e-4 \
    --optimizer_steps 4235 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_steps 10 \
    --max_lr_schedule_steps 4235 \
    --eval_interval 250 \
    --eval_iters 50 \
    --stride 1024 \
    --drop_last \
    --num_workers 4

stop_gpu_monitor

echo "===== RUN COMPLETE ====="
echo "UTC end: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Training log: $RUN_LOG"
if [[ -f "$GPU_LOG" ]]; then
    echo "GPU metrics: $GPU_LOG"
fi