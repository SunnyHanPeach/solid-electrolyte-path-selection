#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != endpoint && "$1" != peak ) ]]; then
    echo "Usage: bash run_phase_b4_spin_gpu.sh endpoint|peak /absolute/path/to/gpu_runtime" >&2
    exit 2
fi
stage="$1"
runtime="$(cd -- "$2" && pwd)"
bundle="$(cd -- "$(dirname -- "$0")" && pwd)"
cd -- "$bundle"
sha256sum --check --strict checksums.sha256

pw="$runtime/usr/local/qe/bin/pw.x"
printf '%s  %s\n' 9d4510c4aff6c7340725bdb9244aef5b6011fdcd91f6a7197f526bdb1f0cc702 "$pw" |
    sha256sum --check --strict

case "$stage" in
    peak)
        for branch in nspin1 spin1muB; do
            if [[ ! -f "runs/endpoint_$branch/converged" ]]; then
                echo "Both endpoint calculations must converge before the peak screen." >&2
                exit 1
            fi
        done
        ;;
esac

unset DISPLAY XAUTHORITY
export LD_LIBRARY_PATH="$runtime/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:$runtime/usr/local/openmpi/lib:$runtime/usr/local/ucx/lib:$runtime/usr/local/fftw/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$runtime/usr/local/openmpi"
export OMPI_MCA_ess_singleton_isolated=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self
export UCX_MEMTYPE_CACHE=n HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" NVCOMPILER_FIO_UNBUFFERED=1

mkdir -p runs
for branch in nspin1 spin1muB; do
    job="${stage}_${branch}"
    if [[ -f "runs/$job/converged" ]]; then
        echo "$job already converged"
        continue
    fi
    (
        mkdir -- "runs/$job"
        cd -- "runs/$job"
        mkdir scratch
        ln -s ../../pseudo pseudo
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
            --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop=2 \
            > gpu_samples.csv &
        sampler=$!
        trap 'kill "$sampler" 2>/dev/null || true' EXIT

        printf 'start_utc=%s\n' "$(date -u +%FT%TZ)" > runtime.txt
        SECONDS=0
        status=0
        "$pw" -in "$bundle/$job.in" > pw.out 2>&1 || status=$?
        printf 'elapsed_seconds=%s\nexit_status=%s\n' "$SECONDS" "$status" >> runtime.txt
        if (( status != 0 )) ||
            ! grep -qi 'GPU acceleration is ACTIVE' pw.out ||
            ! grep -q 'convergence has been achieved' pw.out ||
            ! grep -q 'JOB DONE[.]' pw.out; then
            echo "$job failed validation. See runs/$job/pw.out" >&2
            exit 1
        fi
        touch converged
    )
done
