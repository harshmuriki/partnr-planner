#!/bin/bash
# One array task = one A40 = one shard of the queue, replayed sequentially.
# Submitted by scripts/skynet/submit_gt_timing.sh, which sets --array, --partition, --qos,
# --account, --time and exports REPO, OUT, HABITAT_PYTHON. Do not sbatch this file directly.
#SBATCH --job-name=gt-timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=15
#SBATCH --mem=48G
#SBATCH --requeue

set -euo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=TRUE MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset DISPLAY  # headless EGL rendering on the GPU Slurm assigned (CUDA_VISIBLE_DEVICES is Slurm's; leave it)

shard="$SLURM_ARRAY_TASK_ID"
mkdir -p "$OUT/gpu"
# GPU utilization every 2 s, to tell GPU-bound from CPU-bound actions afterwards.
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 2 > "$OUT/gpu/shard_${shard}.csv" 2>/dev/null &
sampler=$!
trap 'kill $sampler 2>/dev/null || true' EXIT

srun -u "$HABITAT_PYTHON" -u scripts/skynet/gt_timing_replay.py run --out "$OUT" --shard "$shard"
