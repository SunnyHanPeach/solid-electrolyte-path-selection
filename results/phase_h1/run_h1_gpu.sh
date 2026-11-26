#!/usr/bin/env bash
set -eu
py=/root/autodl-tmp/project1-mace-ef/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
while [[ "$mem" -gt 9500 ]]; do
    echo "Waiting for GPU memory. Current usage: $mem MiB"
    sleep 15
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
done
echo "Starting H1 with $mem MiB used on GPU and limited PyTorch allocation."
"$py" phase_h1_gpu.py data protocol.json gpu_output
