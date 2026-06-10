#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/compat/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

for p in /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
  [ -f "$p" ] || continue
  ln -sf "$p" /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null || true
  mkdir -p /usr/local/cuda/compat/lib
  ln -sf "$p" /usr/local/cuda/compat/lib/libcuda.so.1 2>/dev/null || true
  ldconfig || true
  break
done

RANK="${RANK:-0}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_cache/rank_${RANK}"
export TRITON_CACHE_DIR="/tmp/triton_cache/rank_${RANK}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" || true

NPROC="$(python3 -c 'import torch; c=torch.cuda.device_count(); print(c if c>0 else 1)')"

if [ "${NESTER_ENTRYPOINT_MODE:-training}" = "bench_eval" ]; then
  exec torchrun --nproc_per_node="$NPROC" --nnodes=1 run/cloud/eval/worker.py
fi

exec torchrun --nproc_per_node="$NPROC" --nnodes=1 run/cloud/train/worker.py
