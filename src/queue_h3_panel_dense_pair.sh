#!/usr/bin/env bash
# Keep the WSL caller in the foreground so the QE job survives the handoff.
set -euo pipefail
src=$(dirname -- "$(realpath -- "$0")")
case_dir=$(realpath -- "$1")
spin_audit=$(realpath -- "$2")
seed=$(realpath -- "$3")
middle=$(realpath -- "$4")
cd "$case_dir"
exec 9>>.dense_pair.lock
flock -n 9
test ! -e baseline_validation
test ! -e dense_pair_01
test -s "$middle/base_middle/start.txt"
echo "Waiting for the baseline midpoint receipt"
while [[ ! -s "$middle/started/completion" ]]; do
    sleep 20
    if [[ ! -s "$middle/started/completion" ]] && ! pgrep -x pw.x >/dev/null; then
        echo "The baseline solver exited without completing the receipt" >&2
        exit 1
    fi
done
grep -Fxq qualification_batch_exit=0 "$middle/started/completion"
python3 "$src/audit_h3_panel_anchor_checks.py" baseline "$case_dir" "$spin_audit" "$seed" \
    "$case_dir/baseline_validation" --middle-source "$middle"
date -u +%FT%TZ > baseline_validation/accepted_utc.txt
if pgrep -x pw.x >/dev/null; then
    echo "Another QE calculation is active on the local CPU" >&2
    exit 1
fi

mkdir dense_pair_01
cp -p -- input_manifest.json dense_pair_01/
cp -pr -- pseudo dense_pair_01/
for branch in k_dense_a k_dense_middle; do
    mkdir "dense_pair_01/$branch"
    cp -p -- "$branch/pw.in" "dense_pair_01/$branch/"
done
cd dense_pair_01
export QE_RANKS=8 QE_POOLS=2 QE_CONCURRENT_JOBS=1
echo "Baseline accepted. Running the denser-grid pair"
bash "$src/run_h3_panel_scf_cpu.sh" run k_dense_a k_dense_middle
python3 "$src/audit_h3_panel_anchor_checks.py" dense-pair "$case_dir" "$spin_audit" "$seed" \
    "$case_dir/dense_pair_validation" --dense-source "$PWD"
