#!/usr/bin/env bash
set -eu
py=/root/autodl-tmp/project1-mace-ef/.venv/bin/python
export PYTHONPATH="$PWD/deps" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while [[ ! -s gpu_output/gpu_manifest.json ]]; do
    status="$(ps -p 5628 -o stat= || true)"
    if [[ -z "$status" || "$status" == Z* ]]; then
        echo "GPU verification did not finish."
        exit 1
    fi
    sleep 5
done
"$py" phase_h1_shortlist.py data gpu_output protocol.json structure_output --workers 8
