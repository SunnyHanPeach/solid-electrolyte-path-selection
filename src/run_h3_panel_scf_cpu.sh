#!/usr/bin/env bash
# Run from the prepared H3 single-point directory. Use independent scratch per branch.
set -euo pipefail

mode=${1:?Expected dry or run}
shift
ranks=${QE_RANKS:-8}
pools=${QE_POOLS:-2}
parallel_jobs=${QE_CONCURRENT_JOBS:-1}
branches=(neutral_fixed neutral_local negative)
if (( $# > 0 )); then
    branches=("$@")
    [[ "$parallel_jobs" == 1 ]]
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

[[ $mode == dry || $mode == run ]]
[[ $parallel_jobs == 1 || $parallel_jobs == 2 ]]
test "$((ranks % pools))" -eq 0
test "$((parallel_jobs * ranks))" -le "$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
test -s input_manifest.json

if [[ "$mode" == dry ]]; then
    for branch in "${branches[@]}"; do
        (
            cd "$branch"
            sed -e '/^&CONTROL/a\  nstep = 0,' -e 's|./scratch|./dry_scratch|' pw.in > dry.in
            mpirun -np "$ranks" --bind-to none pw.x -nk "$pools" -in dry.in > dry.log 2>&1
            grep -Fq 'JOB DONE.' dry.log
            if grep -Fq 'iteration #' dry.log; then
                echo "Electronic iterations started in dry mode for $branch" >&2
                exit 1
            fi
            printf '\n%s\n' "$branch"
            grep -E 'Program PWSCF|number of electrons|number of k points|RAM|JOB DONE' dry.log
        )
    done
    exit 0
fi

for branch in "${branches[@]}"; do
    test ! -e "$branch/pw.log"
done
python3 - "${branches[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

m = json.loads(Path("input_manifest.json").read_bytes())
selected = sys.argv[1:]
if len(set(selected)) != len(selected) or set(selected) - m["branches"].keys():
    raise SystemExit("The requested calculation list does not match the input manifest")
entries = [(Path(name) / "pw.in", m["branches"][name]["input_sha256"]) for name in selected]
entries += [(Path("pseudo") / rec["filename"], rec["sha256"]) for rec in m["pseudopotentials"].values()]
for path, expected in entries:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Input changed since preparation: {path}")
PY
mkdir started
printf 'start_utc=%s\nranks=%s\npools=%s\nconcurrent_jobs=%s\n' \
    "$(date -u +%FT%TZ)" "$ranks" "$pools" "$parallel_jobs" > started/receipt
command -v pw.x >> started/receipt
sha256sum "$(command -v pw.x)" >> started/receipt

solve() (
    cd "$1"
    printf 'start_utc=%s\n' "$(date -u +%FT%TZ)" > start.txt
    SECONDS=0
    status=0
    /usr/bin/time -v -o runtime.txt mpirun -np "$ranks" --bind-to none pw.x -nk "$pools" -in pw.in > pw.log 2>&1 || status=$?
    printf 'end_utc=%s\nelapsed_seconds=%s\nsolver_exit=%s\n' "$(date -u +%FT%TZ)" "$SECONDS" "$status" > end.txt
    test "$status" -eq 0 &&
        grep -Fq 'JOB DONE.' pw.log &&
        grep -Fq 'convergence has been achieved' pw.log &&
        test -s "scratch/$1.save/data-file-schema.xml"
)

status=0
if [[ $parallel_jobs == 2 ]]; then
    solve neutral_fixed & fixed=$!
    solve neutral_local & local_spin=$!
    wait "$fixed" || status=1
    wait "$local_spin" || status=1
    solve negative || status=1
else
    for branch in "${branches[@]}"; do
        solve "$branch" || status=1
    done
fi
printf 'end_utc=%s\nqualification_batch_exit=%s\n' "$(date -u +%FT%TZ)" "$status" > started/completion
exit "$status"
