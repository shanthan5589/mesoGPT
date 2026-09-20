#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere with:
# bash runs/exp_003.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ ! -x /usr/bin/time ]]; then
    echo "/usr/bin/time not found. Run: sudo apt update && sudo apt install -y time" >&2
    exit 1
fi

RUN_TIMESTAMP="$(date -u +'%Y-%m-%d_%H-%M-%S_UTC')"
RUNS_DIR="$PROJECT_ROOT/outputs/exp_003"
RUN_DIR="$RUNS_DIR/$RUN_TIMESTAMP"

mkdir -p "$RUN_DIR"

RUN_LOG="$RUN_DIR/train.log"
GPU_LOG="$RUN_DIR/gpu_metrics.csv"

exec > >(tee "$RUN_LOG") 2>&1

VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Virtual environment not found. Create .venv and run bash setup.sh cpu or gpu (see README.md)." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

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

echo "===== EXPERIMENT START ====="
run_experiment() {
    local id="$1"
    local micro_batch_size="$2"
    local profiling_enabled="$3"
    local performance_timer="$4"

    local run_dir="$RUN_DIR/ID_${id}"
    mkdir -p "$run_dir"

    echo "===== RUN-${id}: micro-batch ${micro_batch_size}, profiling: ${profiling_enabled}, performance timer: ${performance_timer} ====="

    /usr/bin/time -v python -u src/mesoGPT/training/trainer.py \
        --context_length 1024 \
        --n_embed 768 \
        --n_layers 12 \
        --n_heads 12 \
        --dropout 0.1 \
        --global_batch_size 512 \
        --micro_batch_size "$micro_batch_size" \
        --optimizer_steps 40 \
        --max_lr 3e-4 \
        --min_lr 3e-5 \
        --warmup_steps 10 \
        --max_lr_schedule_steps 40 \
        --eval_interval 10 \
        --eval_iters 64 \
        --stride 1024 \
        --num_workers 4 \
        --run_dir "$run_dir" \
        --profiling_enabled "$profiling_enabled" \
        --performance_timer "$performance_timer" \
        --tokenizer-dir "$PROJECT_ROOT/outputs/exp_001/2026-08-24_18-58-31_UTC" \
        --tokenizer-name "tok-v16384-c450.00m" \
        --attention "sdpa"
}

run_experiment 1 16 False True
run_experiment 2 32 False True
run_experiment 3 64 False True
run_experiment 4 16 True False
run_experiment 5 32 True False
run_experiment 6 64 True False


stop_gpu_monitor

echo "===== RUN COMPLETE ====="
echo "UTC end: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Run directory: $RUN_DIR"
echo "Training log: $RUN_LOG"

if [[ -f "$GPU_LOG" ]]; then
    echo "GPU metrics: $GPU_LOG"
fi