#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
run="$PWD"
baseline=/root/autodl-tmp/b4-r2-production-12341/b4-r2-bundle-12341
runtime=/root/autodl-tmp/b4_bridge04/gpu_runtime
reference=/root/b4-r2-scratch-12341/io_k222
a_scratch=/root/mp1191903-ci-pilot-a
b_scratch="$run/work_b"
resume_a=false
if [[ ${1:-} == --resume-a ]]; then
    resume_a=true
elif (( $# != 0 )); then
    exit 2
fi
exec 9>.active.lock
flock -n 9
if "$resume_a"; then
    test -s endpoint_queue_status.txt
    grep -Fq 'Maximum CPU time exceeded' "$a_scratch/endpoint_a/pw.out"
    grep -Fq 'JOB DONE.' "$a_scratch/endpoint_a/pw.out"
    test ! -e endpoint_a_receipt_window01
    test ! -e work_b/endpoint_b
    mv -- endpoint_queue_status.txt endpoint_queue_status_window01.txt
    if [[ -e path_smoke_status.txt ]]; then
        mv -- path_smoke_status.txt path_smoke_status_window01.txt
    fi
fi
trap 'status=$?; printf "exit_status=%s\nend_utc=%s\n" "$status" "$(date -u +%FT%TZ)" > endpoint_queue_status.txt' EXIT
sha256sum --check --strict prepared_inputs.sha256
(cd "$baseline" && sha256sum --check --strict checksums.sha256)
(cd /root/b4-r2-scratch-12341 && sha256sum --check --strict io_k222/final_density.sha256)
nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used,driver_version --format=csv > gpu.csv
gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
(( gpu_free >= 22000 ))
if ! "$resume_a"; then
    mkdir "$a_scratch" "$b_scratch"
    ln -s "$a_scratch" work_a
fi

restart_endpoint_a() {
    local current=$1
    local origin=/root/mp1191903-ci-pilot-a/endpoint_a
    local prefix=b4_mp1191903_endpoint_spin1muB
    local pw="$runtime/usr/local/qe/bin/pw.x"
    local available
    for f in "$prefix.bfgs" "$prefix.restart_k" "$prefix.restart_scf" "$prefix.wfc1" "$prefix.mix1" "$prefix.update" "$prefix.save/data-file-schema.xml" "$prefix.save/charge-density.dat"; do
        test -s "$origin/scratch/$f" || return 1
    done
    available="$(df -Pm -- /root | awk 'END {print $4}')"
    (( available >= 16384 )) || return 1
    mkdir "$current" || return 1
    cp -a --reflink=auto "$origin/scratch" "$current/" || return 1
    cmp "$origin/scratch/$prefix.wfc1" "$current/scratch/$prefix.wfc1" || return 1
    sed "s/restart_mode = 'from_scratch'/restart_mode = 'restart'/" "$run/endpoint_a.in" > "$current/endpoint_a.in" || return 1
    grep -Fq "restart_mode = 'restart'" "$current/endpoint_a.in" || return 1
    ln -s "$baseline/pseudo" "$current/pseudo" || return 1
    mv -- "$run/endpoint_a_receipt" "$run/endpoint_a_receipt_window01" || return 1
    mv -- "$run/work_a" "$run/work_a_window01" || return 1
    ln -s "$(dirname "$current")" "$run/work_a" || return 1
    printf '%s  %s\n' 9d4510c4aff6c7340725bdb9244aef5b6011fdcd91f6a7197f526bdb1f0cc702 "$pw" |
        sha256sum --check --strict || return 1
    cd "$current" || return 1
    sha256sum endpoint_a.in > input.sha256 || return 1
    unset DISPLAY XAUTHORITY
    export LD_LIBRARY_PATH="$runtime/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:$runtime/usr/local/openmpi/lib:$runtime/usr/local/ucx/lib:$runtime/usr/local/fftw/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OPAL_PREFIX="$runtime/usr/local/openmpi"
    export OMPI_MCA_ess_singleton_isolated=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self
    export UCX_MEMTYPE_CACHE=n HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
    export CUDA_VISIBLE_DEVICES=0 NVCOMPILER_FIO_UNBUFFERED=1
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop=2 > gpu_samples.csv &
    local sampler=$! status=0
    trap "kill $sampler 2>/dev/null || true" EXIT
    printf 'start_utc=%s\n' "$(date -u +%FT%TZ)" > runtime.txt
    SECONDS=0
    "$pw" -in "$current/endpoint_a.in" > pw.out 2>&1 || status=$?
    printf 'elapsed_seconds=%s\nexit_status=%s\n' "$SECONDS" "$status" >> runtime.txt
    (( status == 0 )) && grep -Fq 'GPU acceleration is ACTIVE' pw.out &&
        grep -Fq 'convergence has been achieved' pw.out && grep -Fq 'JOB DONE.' pw.out &&
        ! grep -Fq 'Maximum CPU time exceeded' pw.out
}

run_endpoint() {
    local role=$1 work=$2 min_mib=$3
    shift 3
    local space status=0 current save
    space="$(df -Pm -- "$work" | awk 'END {print $4}')"
    (( space >= min_mib ))
    current="$work/$role"
    printf '%s %s from %s free_mib=%s\n' "$(date -Is)" "$role" "$work" "$space"
    if [[ "$role" == endpoint_a ]] && "$resume_a"; then
        (set -e; restart_endpoint_a "$current") || status=$?
    else
        (cd "$work" && bash "$baseline/run_phase_b4_numeric_gpu.sh" "$run/$role.in" "$runtime" "$baseline/pseudo" "$@") || status=$?
    fi
    local receipt="$run/${role}_receipt"
    mkdir "$receipt"
    for f in pw.out runtime.txt gpu_samples.csv input.sha256 starting_density.sha256; do
        if [[ -s "$current/$f" ]]; then
            cp "$current/$f" "$receipt/"
        fi
    done
    save="$current/scratch/b4_mp1191903_endpoint_spin1muB.save"
    if [[ -s "$save/data-file-schema.xml" ]]; then
        cp "$save/data-file-schema.xml" "$receipt/"
    fi
    if (( status == 0 )); then
        /root/miniconda3/bin/python - "$receipt/data-file-schema.xml" <<'PY'
import math
import sys
from xml.etree import ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
scf = root.find("output/convergence_info/scf_conv")
ions = root.find("output/convergence_info/opt_conv")
forces = root.find("output/forces")
if (root.find("closed") is None or scf.findtext("convergence_achieved") != "true"
        or float(scf.findtext("scf_error")) * 2 > 1e-8
        or ions is None or ions.findtext("convergence_achieved") != "true"
        or forces.attrib.get("dims", "").split() != ["3", "87"]):
    raise SystemExit("Endpoint convergence or full force output failed")
values = list(map(float, forces.text.split()))
maximum = max(math.sqrt(sum(x*x for x in values[i:i+3])) for i in range(0, len(values), 3))*51.42206748
if len(values) != 261 or not all(map(math.isfinite, values)) or maximum > 0.05:
    raise SystemExit("Endpoint force norm exceeds 0.05 eV/Å")
print(f"Maximum endpoint force {maximum:.6f} eV/Å")
PY
        (cd "$receipt" && sha256sum -- pw.out data-file-schema.xml > output.sha256)
        touch "$receipt/validated"
    fi
    return "$status"
}

density="$reference/scratch/b4_mp1191903_endpoint_spin1muB.save/charge-density.dat"
if "$resume_a"; then
    fresh=/root/mp1191903-ci-pilot-a-resume01
    mkdir "$fresh"
    run_endpoint endpoint_a "$fresh" 16384
else
    run_endpoint endpoint_a "$a_scratch" 24576 "$density"
fi
run_endpoint endpoint_b "$b_scratch" 16384
