#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
run="$PWD"
baseline=/root/autodl-tmp/b4-r2-production-12341/b4-r2-bundle-12341
runtime=/root/autodl-tmp/b4_bridge04/gpu_runtime
reference=/root/b4-r2-scratch-12341/io_k222
a_scratch=/root/mp1191903-ci-pilot-a
b_scratch="$run/work_b"
exec 9>.active.lock
flock -n 9
trap 'status=$?; printf "exit_status=%s\nend_utc=%s\n" "$status" "$(date -u +%FT%TZ)" > endpoint_queue_status.txt' EXIT
sha256sum --check --strict prepared_inputs.sha256
(cd "$baseline" && sha256sum --check --strict checksums.sha256)
(cd /root/b4-r2-scratch-12341 && sha256sum --check --strict io_k222/final_density.sha256)
nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used,driver_version --format=csv > gpu.csv
gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
(( gpu_free >= 22000 ))
mkdir "$a_scratch" "$b_scratch"
ln -s "$a_scratch" work_a

run_endpoint() {
    local role=$1 work=$2 min_mib=$3
    shift 3
    local space status=0 current save
    space="$(df -Pm -- "$work" | awk 'END {print $4}')"
    (( space >= min_mib ))
    current="$work/$role"
    printf '%s %s from %s free_mib=%s\n' "$(date -Is)" "$role" "$work" "$space"
    (cd "$work" && bash "$baseline/run_phase_b4_numeric_gpu.sh" "$run/$role.in" "$runtime" "$baseline/pseudo" "$@") || status=$?
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
run_endpoint endpoint_a "$a_scratch" 24576 "$density"
run_endpoint endpoint_b "$b_scratch" 16384
