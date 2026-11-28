#!/usr/bin/env bash
# Continue the frozen five-image ordinary NEB after the existing endpoint and smoke queue.
set -euo pipefail

run=/root/autodl-tmp/mp1191903-ci-pilot-20261127
python=/root/autodl-tmp/project1-mace-ef/.venv/bin/python
runtime=/root/autodl-tmp/b4_bridge04/gpu_runtime
baseline=/root/autodl-tmp/b4-r2-production-12341/b4-r2-bundle-12341
cd "$run"
status_dir="$run/path5_production"
mkdir "$status_dir"
trap 'status=$?; printf "exit_status=%s\nend_utc=%s\n" "$status" "$(date -u +%FT%TZ)" > "$status_dir/status.txt"' EXIT

marker=no_8h_continuation/continuation_status.txt
while [[ ! -s "$marker" ]]; do
    if ! ps -C bash -o args= | grep -E '(^|[ /])continue_qe_a_without_8h\.sh --wait-for-checkpoint$' >/dev/null; then
        echo 'The existing endpoint/short-test queue exited without a final receipt.' >&2
        exit 1
    fi
    sleep 60
done

# Keep the same GPU lock as the endpoint and smoke jobs.
exec 9>>.active.lock
flock 9
for receipt in "$marker" endpoint_queue_status.txt path_smoke_status.txt; do
    grep -Fxq exit_status=0 "$receipt"
done
for endpoint in endpoint_a endpoint_b; do
    test -f "${endpoint}_receipt/validated"
    (cd "${endpoint}_receipt" && sha256sum -c --strict output.sha256)
done
sha256sum -c --strict prepared_inputs.sha256
sha256sum -c --strict path_seed.sha256
(cd "$baseline" && sha256sum -c --strict checksums.sha256)

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
probe=path5_probe
cp -p -- "$probe/block_result.json" "$status_dir/smoke_result.json"
cp -p -- "$probe/protocol.json" "$status_dir/smoke_protocol.json"
"$python" - "$probe" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

from ase.io import read
import numpy as np

p = Path(sys.argv[1])
latest = p / "latest.traj"
result = json.loads((p / "block_result.json").read_bytes())
protocol = json.loads((p / "protocol.json").read_bytes())
images = read(latest, ":")
if (hashlib.sha256(latest.read_bytes()).hexdigest() != result["latest_sha256"]
    or protocol["climb"] or protocol["fmax"] != 0.10 or len(images) != 5
    or not all(math.isfinite(result[k]) for k in ("fmax", "forward_eV", "reverse_eV"))
    or any(len(a) != 87 or a.get_forces().shape != (87, 3)
           or not np.isfinite(a.get_forces()).all() for a in images)):
    raise SystemExit("Invalid five-image ordinary NEB smoke checkpoint")
if (p / "passed").exists() != (result["converged"] and result["fmax"] <= 0.10):
    raise SystemExit("The smoke checkpoint and convergence marker disagree")
PY

if [[ -f "$probe/passed" ]]; then
    echo 'The five-image smoke checkpoint already meets the ordinary NEB threshold.'
    cp -p -- "$probe/block_result.json" "$status_dir/final_result.json"
    exit 0
fi

gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
(( gpu_free >= 22000 ))
(( $(df -Pm -- "$run" | awk 'END {print $4}') >= 4096 ))
args=(path5_seed.traj "$baseline/io_k222.in" "$runtime" "$baseline/pseudo" "$probe" --fmax 0.10)
printf 'start_utc=%s\n' "$(date -u +%FT%TZ)" > "$status_dir/timing.txt"
SECONDS=0
result=0
"$python" run_mp1191903_ase_neb.py "${args[@]}" --steps 12 || result=$?
printf 'elapsed_seconds=%s\nsolver_exit_status=%s\n' "$SECONDS" "$result" >> "$status_dir/timing.txt"
test "$result" -eq 0
cp -p -- "$probe/block_result.json" "$status_dir/final_result.json"
if [[ ! -f "$probe/passed" ]]; then
    echo 'Ordinary NEB checkpoint saved; convergence still pending.'
    exit 2
fi
echo 'Five-image ordinary NEB meets the frozen force threshold.'
