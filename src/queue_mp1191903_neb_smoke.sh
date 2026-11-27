#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
run="$PWD"
python=/root/autodl-tmp/project1-mace-ef/.venv/bin/python
runtime=/root/autodl-tmp/b4_bridge04/gpu_runtime
baseline=/root/autodl-tmp/b4-r2-production-12341/b4-r2-bundle-12341
exec 9>.active.lock
flock 9
trap 'status=$?; printf "exit_status=%s\nend_utc=%s\n" "$status" "$(date -u +%FT%TZ)" > path_smoke_status.txt' EXIT
grep -Fxq exit_status=0 endpoint_queue_status.txt
for end in endpoint_a endpoint_b; do
    test -f "${end}_receipt/validated"
    (cd "${end}_receipt" && sha256sum -c --strict output.sha256)
done
sha256sum -c --strict path_seed.sha256
(cd "$baseline" && sha256sum -c --strict checksums.sha256)
gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
(( gpu_free >= 22000 ))
(( $(df -Pm -- "$run" | awk 'END {print $4}') >= 4096 ))
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
endpoint_xml=(endpoint_a_receipt/data-file-schema.xml endpoint_b_receipt/data-file-schema.xml)
"$python" prepare_mp1191903_neb.py path5_seed.traj five seed.extxyz "${endpoint_xml[@]}"
args=(path5_seed.traj "$baseline/io_k222.in" "$runtime" "$baseline/pseudo" path5_probe --fmax 0.10)
"$python" run_mp1191903_ase_neb.py "${args[@]}" --steps 0
if [[ ! -f path5_probe/passed ]]; then
    "$python" run_mp1191903_ase_neb.py "${args[@]}" --steps 1
fi
cat path5_probe/block_result.json
