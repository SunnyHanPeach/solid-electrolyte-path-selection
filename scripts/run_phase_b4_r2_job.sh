#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
    echo "Usage: bash run_phase_b4_r2_job.sh bundle runtime run_dir job [starting_density]" >&2
    exit 2
fi
bundle="$(realpath "$1")"
runtime="$(realpath "$2")"
run="$(realpath "$3")"
job="$4"
[[ "$job" =~ ^(io_k222(_cold)?|(endpoint|peak)_k333(_cold)?|middle_k(222|333))$ ]]
(cd "$bundle" && sha256sum --check --strict checksums.sha256)
test -d "$run"
input="$bundle/$job.in"
test -s "$input"

options=()
if grep -Eq "startingpot[[:space:]]*=[[:space:]]*'file'" "$input"; then
    if [[ $# -ne 5 || ! -s "$5" ]]; then
        echo "A readable starting density is required for $job." >&2
        exit 2
    fi
    options+=("$(realpath "$5")")
elif [[ $# -ne 4 ]]; then
    echo "A density was supplied for a cold-start input." >&2
    exit 2
fi

cd "$run"
exec 9>.single_gpu.lock
flock -n 9 || { echo "A B4-R2 job is already running in $run" >&2; exit 2; }
if [[ -e "$job" || -e "$job.results.tar.gz" ]]; then
    echo "Preserving existing results for $job in $run. Use a fresh run directory for a retry." >&2
    exit 2
fi
device="${CUDA_VISIBLE_DEVICES:-0}"
memory="$(nvidia-smi -i "$device" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits)"
if [[ ! "$memory" =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]]; then
    echo "Unexpected GPU memory response: $memory" >&2
    exit 2
fi
total="${BASH_REMATCH[1]}"
used="${BASH_REMATCH[2]}"
available_mib="$(df -Pm -- "$run" | awk 'END {print $4}')"
if (( total - used < 22000 || available_mib < 24576 )); then
    echo "Insufficient available VRAM or scratch. Free MiB: GPU $((total-used)), disk $available_mib" >&2
    exit 2
fi

nvidia-smi -i "$device" --query-gpu=name,memory.total,driver_version --format=csv > "$job.hardware.csv"
df -Pm -- "$run" > "$job.disk_before.txt"
status=0
bash "$bundle/run_phase_b4_numeric_gpu.sh" "$input" "$runtime" "$bundle/pseudo" "${options[@]}" || status=$?
if [[ ! -d "$job" ]]; then
    exit "$status"
fi
prefix="$(awk -F"'" '/^[[:space:]]*prefix[[:space:]]*=/ { print $2 }' "$input")"
[[ "$prefix" =~ ^[[:alnum:]_-]+$ ]]
save="$job/scratch/$prefix.save"
if [[ -s "$save/data-file-schema.xml" ]]; then
    cp -- "$save/data-file-schema.xml" "$job/data-file-schema.xml"
    sha256sum "$job/data-file-schema.xml" > "$job/xml.sha256"
fi
if [[ -s "$save/charge-density.dat" ]]; then
    sha256sum "$save/charge-density.dat" > "$job/final_density.sha256"
fi
df -Pm -- "$run" > "$job.disk_after.txt"
du -sm -- "$job/scratch" > "$job.scratch_size.txt"
files=("$job.hardware.csv" "$job.disk_before.txt" "$job.disk_after.txt" "$job.scratch_size.txt")
for basename in pw.out runtime.txt gpu_samples.csv converged input.sha256 starting_density.sha256 \
    data-file-schema.xml xml.sha256 final_density.sha256; do
    if [[ -f "$job/$basename" ]]; then files+=("$job/$basename"); fi
done
tar -czf "$job.results.tar.gz" "${files[@]}"
sha256sum "$job.results.tar.gz" > "$job.results.sha256"
if (( status == 0 )); then
    test -s "$job/data-file-schema.xml"
    test -s "$job/final_density.sha256"
fi
exit "$status"
