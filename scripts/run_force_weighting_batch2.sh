#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 PROTECTED_BUNDLE.zip OUTPUT" >&2
    exit 2
fi

source_dir=$(cd -- "$(dirname -- "$0")/../src" && pwd)
bundle=$1
output=$2
python=${PYTHON_BIN:-python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$output"

status() {
    printf "%s %s\n" "$1" "$(date -u +%FT%TZ)" > "$output/state.tmp"
    mv -- "$output/state.tmp" "$output/state"
}
trap 'rc=$?; if ((rc)); then status "error:$rc"; fi' EXIT
status checking
"$python" -c 'import torch, mace, ase, numpy; assert torch.cuda.is_available(); assert (torch.__version__, mace.__version__, ase.__version__, numpy.__version__) == ("2.8.0+cu128", "0.3.14", "3.29.0", "2.2.6")'
for seed in 17 29; do
    "$python" "$source_dir/run_force_weighting.py" "$bundle" "$output" --replicate-batch2 --seed "$seed" --preflight > "$output/design_seed$seed.json"
done

run() {
    local arm=$1 seed=$2
    "$python" -u "$source_dir/run_force_weighting.py" "$bundle" "$output" --replicate-batch2 --seed "$seed" --arm "$arm" --device cuda >> "$output/${arm}_seed$seed.log" 2>&1
}
for arm in replay full uniform_weighted energy_weighted; do
    status "training:$arm"
    run "$arm" 17 &
    a=$!
    run "$arm" 29 &
    b=$!
    error=0
    wait "$a" || error=1
    wait "$b" || error=1
    if ((error)); then
        echo "Retrying $arm sequentially"
        run "$arm" 17
        run "$arm" 29
    fi
done

for arm in replay full uniform_weighted energy_weighted; do
    for seed in 17 29; do
        test -s "$output/seed$seed/$arm/done_807.json"
    done
done
tar -C "$output" --exclude='model_*.pt' -czf "$output/compact.tar.gz.partial" seed17 seed29 design_seed17.json design_seed29.json
mv "$output/compact.tar.gz.partial" "$output/compact.tar.gz"
(cd -- "$output" && sha256sum compact.tar.gz > compact.sha256)
status complete
