#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
bundle=$1
output=$2
python=${PYTHON_BIN:-python}
mkdir -p "$output"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" > "$output/state.tmp"; mv "$output/state.tmp" "$output/state"; }
trap 'code=$?; if ((code)); then status "error:$code"; fi' EXIT
status smoke
"$python" -u src/phase1_path_value.py run "$bundle" "$output/smoke" --arm path1 --seed 17 --smoke 32 > "$output/smoke.log" 2>&1
status training
run() {
    local arm=$1 seed=$2
    "$python" -u src/phase1_path_value.py run "$bundle" "$output" --arm "$arm" --seed "$seed" >> "$output/${arm}_seed$seed.log" 2>&1
}
tasks=(pure:17 pure:29 pure:43)
for i in 1 2 3; do
    for seed in 17 29 43; do tasks+=("path$i:$seed" "replay$i:$seed"); done
done
for (( i=0;i<${#tasks[@]};i+=2 )); do
    pids=()
    for tag in "${tasks[@]:i:2}"; do run "${tag%:*}" "${tag#*:}" & pids+=("$!"); done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if ((failed)); then
        for tag in "${tasks[@]:i:2}"; do run "${tag%:*}" "${tag#*:}"; done
    fi
done
tar -C "$output" --exclude='model_*.pt' -czf "$output/compact.tar.gz.partial" seed17 seed29 seed43
mv "$output/compact.tar.gz.partial" "$output/compact.tar.gz"
(cd "$output" && sha256sum compact.tar.gz > compact.sha256)
status complete
