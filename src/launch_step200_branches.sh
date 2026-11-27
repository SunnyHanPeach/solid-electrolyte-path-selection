#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
threads="${TRAIN_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum --check inputs.sha256
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > gpu_environment.csv
"$python_bin" -m pip freeze > python_environment.txt
common=(--device cuda --threads "$threads" --branch-protocol continuation_protocol_inputs.json)
run() {
    local output="$1" seed="$2"
    shift 2
    "$python_bin" src/run_adam_cross.py frozen_bundle.zip step200_states.zip "$output" \
        --seed "$seed" "${common[@]}" "$@"
}
run smoke 17 --smoke
run full 17
run full 29
