#!/usr/bin/env bash
# Restore the stopped GPU-QE endpoint B, then continue its frozen five-image work.
set -euo pipefail
stage=$(realpath -- "$1")
mode=$2
project="$stage/project"
run="$project/mp1191903-ci-pilot-20261127"
work="$run/work_b/endpoint_b"
runtime="$project/b4_bridge04/gpu_runtime"
baseline="$project/b4-r2-production-12341/b4-r2-bundle-12341"
prefix=b4_mp1191903_endpoint_spin1muB
pw="$runtime/usr/local/qe/bin/pw.x"
python="$stage/venv/bin/python"
status="$stage/resume"
cd "$run"

export LD_LIBRARY_PATH="$runtime/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:$runtime/usr/local/openmpi/lib:$runtime/usr/local/ucx/lib:$runtime/usr/local/fftw/lib:$runtime/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export OPAL_PREFIX="$runtime/usr/local/openmpi"
export OMPI_MCA_ess_singleton_isolated=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self
export UCX_MEMTYPE_CACHE=n HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 NVCOMPILER_FIO_UNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset DISPLAY XAUTHORITY

if [[ "$mode" == check ]]; then
    [[ -s "$stage/runtime.verified" && -s "$stage/state.verified" && -s "$stage/compat.verified" ]]
    test "$(sha256sum "$pw" | cut -d' ' -f1)" = 9d4510c4aff6c7340725bdb9244aef5b6011fdcd91f6a7197f526bdb1f0cc702
    ldd "$pw" > "$stage/runtime.ldd"
    if grep -F 'not found' "$stage/runtime.ldd"; then
        exit 1
    fi
    (cd "$runtime/usr/local/cuda/lib64" && sha256sum -c --strict "$stage/incoming/compat_files.sha256")
    (cd "$baseline" && sha256sum -c --strict checksums.sha256)
    (cd "$project" && sha256sum -c --strict ../incoming/state_files.sha256)
    (cd endpoint_a_receipt && test -f validated && sha256sum -c --strict output.sha256)
    sha256sum -c --strict prepared_inputs.sha256
    sha256sum -c --strict path_seed.sha256
    grep -Fiq 'stopped by user request' "$work/pw.out"
    grep -Fq 'JOB DONE.' "$work/pw.out"
    test -s "$work/scratch/$prefix.restart_scf"
    test ! -e "$work/$prefix.EXIT"
    test ! -e "$work/scratch/$prefix.EXIT"
    # The transferred absolute symlink names the original provider's data disk.
    test -L "$work/pseudo"
    ln -sfnT -- "$baseline/pseudo" "$work/pseudo"
    mkdir "$status"
    grep -Eq "^ *restart_mode *= *'from_scratch' *$" endpoint_b.in
    sed -E "s/^([[:space:]]*restart_mode[[:space:]]*=[[:space:]]*)'from_scratch'[[:space:]]*$/\1'restart'/" endpoint_b.in > "$status/endpoint_b.restart.in"
    sed -E "s/^([[:space:]]*restart_mode[[:space:]]*=[[:space:]]*)'restart'[[:space:]]*$/\1'from_scratch'/" "$status/endpoint_b.restart.in" | cmp - endpoint_b.in
    sha256sum "$status/endpoint_b.restart.in" > "$status/endpoint_b.restart.sha256"
    date -Is > "$status/preflight.ok"
    exit 0
fi

[[ "$mode" == run && -s "$status/preflight.ok" ]]
exec 9>>.active.lock
flock -n 9
test ! -e "$status/endpoint_b.log"
test ! -e endpoint_b_receipt
sha256sum -c --strict "$status/endpoint_b.restart.sha256"
(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 23500 ))
(( $(df -Pm "$run" | awk 'END {print $4}') >= 4096 ))
trap 'rc=$?; printf "exit_status=%s\nend_utc=%s\n" "$rc" "$(date -u +%FT%TZ)" > "$status/status.txt"' EXIT
date -u +%FT%TZ > "$status/start_utc.txt"
(
    cd "$work"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop=5 > "$status/gpu.csv" &
    gpu_pid=$!
    trap 'kill "$gpu_pid" 2>/dev/null || true' EXIT
    SECONDS=0
    rc=0
    "$pw" -in "$status/endpoint_b.restart.in" > "$status/endpoint_b.log" 2>&1 || rc=$?
    printf 'elapsed_seconds=%s\nsolver_exit_status=%s\n' "$SECONDS" "$rc" > "$status/endpoint_b.timing"
    test "$rc" -eq 0
)
grep -Fq 'GPU acceleration is ACTIVE' "$status/endpoint_b.log"
grep -Fq 'JOB DONE.' "$status/endpoint_b.log"
if grep -Ei 'stopped by user request|maximum cpu time exceeded' "$status/endpoint_b.log"; then
    exit 1
fi

mkdir "$status/endpoint_b_receipt"
cp -- "$status/endpoint_b.log" "$status/endpoint_b_receipt/pw.out"
cp -- "$work/scratch/$prefix.save/data-file-schema.xml" "$status/endpoint_b_receipt/"
python3 - "$status/endpoint_b_receipt/data-file-schema.xml" <<'PY'
import math
import sys
from xml.etree import ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
scf = root.find("output/convergence_info/scf_conv")
ions = root.find("output/convergence_info/opt_conv")
forces = root.find("output/forces")
if (root.find("closed") is None or scf is None or ions is None or forces is None
        or scf.findtext("convergence_achieved") != "true"
        or ions.findtext("convergence_achieved") != "true"
        or 2 * float(scf.findtext("scf_error")) > 1e-8
        or forces.get("dims", "").split() != ["3", "87"]):
    raise SystemExit("Endpoint B did not meet the frozen electronic or ionic gate")
values = [float(x) for x in forces.text.split()]
if len(values) != 261 or not all(map(math.isfinite, values)):
    raise SystemExit("Expected 261 finite force components")
maximum = max(math.sqrt(sum(x*x for x in values[i:i+3])) for i in range(0, 261, 3))*51.42206748
if maximum > 0.05:
    raise SystemExit(f"Endpoint B maximum force {maximum:g} eV/A exceeds the acceptance threshold")
print(f"Endpoint B maximum force {maximum:.8f} eV/A")
PY
(cd "$status/endpoint_b_receipt" && sha256sum -- pw.out data-file-schema.xml > output.sha256 && touch validated)
mv -T -- "$status/endpoint_b_receipt" endpoint_b_receipt
date -Is > "$status/endpoint_b.validated"

test -x "$python"
neb_driver="$stage/run_mp1191903_ase_neb.py"
test -s "$neb_driver"
"$python" prepare_mp1191903_neb.py path5_seed.traj five seed.extxyz \
    endpoint_a_receipt/data-file-schema.xml endpoint_b_receipt/data-file-schema.xml
args=(path5_seed.traj "$baseline/io_k222.in" "$runtime" "$baseline/pseudo" path5_probe --fmax 0.10)
"$python" "$neb_driver" "${args[@]}" --steps 0
if [[ ! -f path5_probe/passed ]]; then
    "$python" "$neb_driver" "${args[@]}" --steps 1
fi
cp -- path5_probe/block_result.json "$status/path5_smoke_result.json"
cp -- path5_probe/protocol.json "$status/path5_smoke_protocol.json"
"$python" - <<'PY'
import hashlib
import json
import math
from pathlib import Path

from ase.io import read
import numpy as np

path = Path("path5_probe")
latest = path / "latest.traj"
result = json.loads((path / "block_result.json").read_bytes())
protocol = json.loads((path / "protocol.json").read_bytes())
images = read(latest, ":")
if (hashlib.sha256(latest.read_bytes()).hexdigest() != result["latest_sha256"]
        or protocol["climb"] or protocol["fmax"] != 0.10 or len(images) != 5
        or not all(math.isfinite(result[key]) for key in ("fmax", "forward_eV", "reverse_eV"))
        or any(len(image) != 87 or image.get_forces().shape != (87, 3)
               or not np.isfinite(image.get_forces()).all() for image in images)):
    raise SystemExit("Invalid five-image ordinary NEB smoke checkpoint")
if (path / "passed").exists() != (result["converged"] and result["fmax"] <= 0.10):
    raise SystemExit("The smoke checkpoint and convergence marker disagree")
PY
if [[ ! -f path5_probe/passed ]]; then
    SECONDS=0
    result=0
    "$python" "$neb_driver" "${args[@]}" --steps 12 || result=$?
    printf 'elapsed_seconds=%s\nsolver_exit_status=%s\n' "$SECONDS" "$result" > "$status/path5.timing"
    test "$result" -eq 0
fi
cp -- path5_probe/block_result.json "$status/path5_result.json"
test -f path5_probe/passed
