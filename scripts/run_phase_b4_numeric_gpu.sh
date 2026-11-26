#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_phase_b4_numeric_gpu.sh input.in /absolute/path/to/gpu_runtime /absolute/path/to/pseudo [charge-density.dat]" >&2
    exit 2
fi
input="$(realpath "$1")"
runtime="$(realpath "$2")"
pseudo="$(realpath "$3")"
pw="$runtime/usr/local/qe/bin/pw.x"
test -s "$input"
test -d "$pseudo"
printf '%s  %s\n' 9d4510c4aff6c7340725bdb9244aef5b6011fdcd91f6a7197f526bdb1f0cc702 "$pw" |
    sha256sum --check --strict

density=""
if [[ $# -eq 4 ]]; then
    density="$(realpath "$4")"
    test -s "$density"
    grep -Eq "startingpot[[:space:]]*=[[:space:]]*'file'" "$input"
fi

job="$(basename "$input" .in)"
mkdir -- "$job"
cd -- "$job"
mkdir scratch
if [[ -n "$density" ]]; then
    prefix="$(awk -F"'" '/^[[:space:]]*prefix[[:space:]]*=/ { print $2 }' "$input")"
    [[ "$prefix" =~ ^[[:alnum:]_-]+$ ]]
    mkdir "scratch/$prefix.save"
    cp -- "$density" "scratch/$prefix.save/charge-density.dat"
    sha256sum "scratch/$prefix.save/charge-density.dat" > starting_density.sha256
fi
ln -s "$pseudo" pseudo
sha256sum "$input" > input.sha256

unset DISPLAY XAUTHORITY
export LD_LIBRARY_PATH="$runtime/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:$runtime/usr/local/openmpi/lib:$runtime/usr/local/ucx/lib:$runtime/usr/local/fftw/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$runtime/usr/local/openmpi"
export OMPI_MCA_ess_singleton_isolated=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self
export UCX_MEMTYPE_CACHE=n HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" NVCOMPILER_FIO_UNBUFFERED=1

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop=2 > gpu_samples.csv &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

printf 'start_utc=%s\n' "$(date -u +%FT%TZ)" > runtime.txt
SECONDS=0
status=0
"$pw" -in "$input" > pw.out 2>&1 || status=$?
printf 'elapsed_seconds=%s\nexit_status=%s\n' "$SECONDS" "$status" >> runtime.txt
if (( status != 0 )) || ! grep -qi 'GPU acceleration is ACTIVE' pw.out ||
    ! grep -q 'convergence has been achieved' pw.out || ! grep -q 'JOB DONE[.]' pw.out; then
    echo "$job failed validation. See $PWD/pw.out" >&2
    exit 1
fi
touch converged
