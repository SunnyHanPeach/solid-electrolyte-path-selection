#!/usr/bin/env bash
# Preserve the running QE segment and continue from its checkpoint only if its old timer fires.
set -euo pipefail
run=/root/autodl-tmp/mp1191903-ci-pilot-20261127
current=/root/mp1191903-ci-pilot-a-resume01/endpoint_a
baseline=/root/autodl-tmp/b4-r2-production-12341/b4-r2-bundle-12341
runtime=/root/autodl-tmp/b4_bridge04/gpu_runtime
pw="$runtime/usr/local/qe/bin/pw.x"
prefix=b4_mp1191903_endpoint_spin1muB
evidence="$run/no_8h_continuation"
cd "$run"

without_window() {
    test "$(grep -cE '^[[:space:]]*max_seconds[[:space:]]*=[[:space:]]*28800[[:space:],]*$' "$1")" -eq 1
    sed -E '/^[[:space:]]*max_seconds[[:space:]]*=/d' "$1" > "$2"
    test "$(( $(wc -l < "$1") - $(wc -l < "$2") ))" -eq 1
    ! grep -qE '^[[:space:]]*max_seconds[[:space:]]*=' "$2"
}

if [[ ${1:-} == --prepare ]]; then
    test ! -e work_b/endpoint_b
    test ! -e "$evidence"
    sha256sum --check --strict prepared_inputs.sha256
    mkdir "$evidence"
    cp -p -- endpoint_b.in prepared_inputs.sha256 "$evidence/"
    without_window endpoint_b.in "$evidence/endpoint_b.no_8h.in"
    without_window "$current/endpoint_a.in" "$evidence/endpoint_a.restart.no_8h.in"
    grep -Eq "restart_mode[[:space:]]*=[[:space:]]*'restart'" "$evidence/endpoint_a.restart.no_8h.in"
    if grep -qE '^[[:space:]]*max_seconds[[:space:]]*=' "$baseline/io_k222.in"; then
        echo 'The NEB template has an explicit time limit and needs review.' >&2
        exit 1
    fi
    test ! -e work_b/endpoint_b
    cp -- "$evidence/endpoint_b.no_8h.in" endpoint_b.in.no_8h.tmp
    mv -T -- endpoint_b.in.no_8h.tmp endpoint_b.in
    sha256sum -- endpoint_a.in endpoint_b.in > prepared_inputs.no_8h.tmp
    mv -T -- prepared_inputs.no_8h.tmp prepared_inputs.sha256
    sha256sum --check --strict prepared_inputs.sha256
    date -Is > "$evidence/prepared_at.txt"
    exit 0
fi

test "${1:-}" = --wait-for-checkpoint
test -s "$evidence/prepared_at.txt"
trap 'status=$?; printf "exit_status=%s\nend_utc=%s\n" "$status" "$(date -u +%FT%TZ)" > "$evidence/continuation_status.txt"' EXIT
wait_wrapper() {
    local pid state
    pid=$(cat "$1")
    while state=$(ps -p "$pid" -o stat=) && [[ -n "$state" && "$state" != Z* ]]; do
        sleep 20
    done
}

echo "$(date -Is) waiting for the existing endpoint and path wrappers"
wait_wrapper endpoint_queue.resume01.pid
wait_wrapper path_smoke_queue.resume01.pid
exec 9>>.active.lock
flock 9
if [[ -f endpoint_a_receipt/validated && -f endpoint_b_receipt/validated ]]; then
    echo "$(date -Is) existing queue validated both endpoints"
    exit 0
fi
# This contingency never retries an unrelated failure or repeats a validated calculation.
test ! -f endpoint_a_receipt/validated
test ! -e work_b/endpoint_b
grep -Fq 'Maximum CPU time exceeded' endpoint_a_receipt/pw.out
grep -Fq 'JOB DONE.' endpoint_a_receipt/pw.out
for entry in "$prefix.bfgs" "$prefix.wfc1" "$prefix.save/data-file-schema.xml" "$prefix.save/charge-density.dat"; do
    test -s "$current/scratch/$entry"
done
test -s "$current/scratch/$prefix.restart_scf" || test -s "$current/scratch/$prefix.restart_k"
sha256sum --check --strict prepared_inputs.sha256
(cd "$baseline" && sha256sum --check --strict checksums.sha256)
for marker in endpoint_queue_status.txt path_smoke_status.txt; do
    if [[ -f "$marker" ]]; then cp -- "$marker" "$evidence/${marker%.txt}_window02.txt"; fi
done

validate_endpoint() {
    local receipt=$1
    /root/miniconda3/bin/python - "$receipt/data-file-schema.xml" <<'PY'
import math
import sys
from xml.etree import ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
scf = root.find("output/convergence_info/scf_conv")
ions = root.find("output/convergence_info/opt_conv")
forces = root.find("output/forces")
if (root.find("closed") is None or scf is None or forces is None
        or scf.findtext("convergence_achieved") != "true"
        or 2 * float(scf.findtext("scf_error")) > 1e-8
        or ions is None or ions.findtext("convergence_achieved") != "true"
        or forces.attrib.get("dims", "").split() != ["3", "87"]):
    raise SystemExit("Endpoint convergence or full force output failed")
values = [float(v) for v in forces.text.split()]
if len(values) != 261 or not all(map(math.isfinite, values)):
    raise SystemExit("Expected 261 finite force components")
maximum = max(math.sqrt(sum(x*x for x in values[i:i+3])) for i in range(0, 261, 3))*51.42206748
if maximum > 0.05:
    raise SystemExit("Endpoint force norm exceeds 0.05 eV/A")
print(f"Maximum endpoint force {maximum:.6f} eV/A")
PY
    (cd "$receipt" && sha256sum -- pw.out data-file-schema.xml > output.sha256)
    touch "$receipt/validated"
}

echo "$(date -Is) verified timed-out A and resume files; starting from the same scratch checkpoint"
mkdir "$evidence/endpoint_a_receipt" "$evidence/endpoint_b_receipt"
export LD_LIBRARY_PATH="$runtime/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:$runtime/usr/local/openmpi/lib:$runtime/usr/local/ucx/lib:$runtime/usr/local/fftw/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$runtime/usr/local/openmpi"
export OMPI_MCA_ess_singleton_isolated=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self
export UCX_MEMTYPE_CACHE=n HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 NVCOMPILER_FIO_UNBUFFERED=1
unset DISPLAY XAUTHORITY
(
    cd "$current"
    receipt="$evidence/endpoint_a_receipt"
    sha256sum "$evidence/endpoint_a.restart.no_8h.in" > "$receipt/input.sha256"
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop=2 > "$receipt/gpu_samples.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    status=0
    printf 'start_utc=%s\n' "$(date -u +%FT%TZ)" > "$receipt/runtime.txt"
    SECONDS=0
    "$pw" -in "$evidence/endpoint_a.restart.no_8h.in" > "$receipt/pw.out" 2>&1 || status=$?
    printf 'elapsed_seconds=%s\nexit_status=%s\n' "$SECONDS" "$status" >> "$receipt/runtime.txt"
    test "$status" -eq 0
    grep -Fq 'JOB DONE.' "$receipt/pw.out"
    ! grep -Fq 'Maximum CPU time exceeded' "$receipt/pw.out"
    cp -- "scratch/$prefix.save/data-file-schema.xml" "$receipt/"
    validate_endpoint "$receipt"
)
mv -T -- endpoint_a_receipt "$evidence/endpoint_a_receipt_window02"
mv -T -- "$evidence/endpoint_a_receipt" endpoint_a_receipt

echo "$(date -Is) endpoint A validated; starting the uncapped B input"
(cd work_b && bash "$baseline/run_phase_b4_numeric_gpu.sh" "$run/endpoint_b.in" "$runtime" "$baseline/pseudo")
b="$evidence/endpoint_b_receipt"
for file in pw.out runtime.txt gpu_samples.csv input.sha256; do
    if [[ -s "work_b/endpoint_b/$file" ]]; then cp -- "work_b/endpoint_b/$file" "$b/"; fi
done
cp -- "work_b/endpoint_b/scratch/$prefix.save/data-file-schema.xml" "$b/"
validate_endpoint "$b"
mv -T -- "$b" endpoint_b_receipt
printf 'exit_status=0\nend_utc=%s\n' "$(date -u +%FT%TZ)" > endpoint_queue_status.txt
flock -u 9
bash "$run/queue_mp1191903_neb_smoke.sh" > "$evidence/path_smoke.log" 2>&1
