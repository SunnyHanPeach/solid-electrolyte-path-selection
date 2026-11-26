#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
    echo "Usage: bash run_phase_b4_r2_cutoff_queue.sh phase scratch runtime middle_queue_pid python" >&2
    exit 2
fi
phase="$(realpath -e "$1")"
scratch="$(realpath -e "$2")"
runtime="$(realpath -e "$3")"
middle_pid="$4"
python="$(realpath -e "$5")"
[[ "$middle_pid" =~ ^[0-9]+$ ]]
queue="$(cd -- "$(dirname -- "$0")" && pwd)"
bundle="$queue/bundle"
cd "$queue"
sha256sum -c --strict checksums.sha256

echo "Waiting for both non-peak grids."
for job in middle_k222 middle_k333; do
    receipt="$phase/${job}_completion.txt"
    while [[ ! -s "$receipt" ]]; do
        state="$(ps -p "$middle_pid" -o stat= || true)"
        if [[ -z "$state" || "$state" =~ ^Z ]]; then
            echo "Non-peak queue stopped before $job produced a completion receipt." >&2
            exit 3
        fi
        sleep 30
    done
    grep -qx 'exit_status=0' "$receipt"
    (cd "$phase" && sha256sum -c --strict "$job.results.sha256")
done

"$python" "$queue/phase_b4_r2.py" score "$bundle" "$scratch" "$queue/grid_scored" \
    endpoint:endpoint_k333 peak:peak_k333 middle_k222:middle_k333
if ! "$python" - "$queue/grid_scored/scores.json" <<'PY'
import json
import sys
verdict = json.load(open(sys.argv[1], encoding="utf-8"))["verdict"]
sys.exit(not all(verdict[key]["pass_gate"] for key in ("peak:peak_k333", "middle_k222:middle_k333")))
PY
then
    echo "Grid acceptance failed. The cutoff jobs were held."
    exit 4
fi

for job in endpoint_cut125 peak_cut125; do
    deadline="${B4_R2_DEADLINE_EPOCH:-}"
    if [[ -n "${B4_R2_DEADLINE_FILE:-}" ]]; then
        read -r deadline < "$B4_R2_DEADLINE_FILE"
    fi
    if [[ -n "$deadline" ]]; then
        [[ "$deadline" =~ ^[0-9]+$ ]]
        # The larger cutoff has not been benchmarked. Require 135 minutes per
        # cold job, increasing that window if the first cold job ran longer.
        reserve_seconds=8100
        if [[ "$job" == peak_cut125 ]]; then
            elapsed="$("$python" - "$scratch/endpoint_cut125/runtime.txt" <<'PY'
import re
import sys
from pathlib import Path

data = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"^elapsed_seconds=(\d+)$", data, re.MULTILINE)
print(int(match.group(1)) if match else 0)
PY
)"
            reserve_seconds=$(( reserve_seconds > elapsed * 5 / 4 + 600 ? reserve_seconds : elapsed * 5 / 4 + 600 ))
        fi
        if (( $(date +%s) + reserve_seconds > deadline )); then
            printf 'held=%s\nreason=insufficient_time_before_deadline\n' "$job" | tee "$queue/${job}_budget_hold.txt"
            exit 0
        fi
    fi
    echo "Starting $job at $(date -u +%FT%TZ)"
    status=0
    bash "$bundle/run_phase_b4_r2_job.sh" "$bundle" "$runtime" "$scratch" "$job" || status=$?
    for file in "$job.results.tar.gz" "$job.results.sha256"; do
        if [[ -s "$scratch/$file" ]]; then
            cp -- "$scratch/$file" "$queue/$file.partial"
            mv -- "$queue/$file.partial" "$queue/$file"
        fi
    done
    printf 'exit_status=%s\nend_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$queue/${job}_completion.txt"
    if (( status != 0 )); then exit "$status"; fi
    sha256sum -c --strict "$job.results.sha256"
    role="${job%_cut125}"
    "$python" "$queue/phase_b4_r2.py" score "$bundle" "$scratch" "$queue/${job}_scored" "$role:$job"
done

"$python" "$queue/phase_b4_r2.py" score "$bundle" "$scratch" "$queue/cutoff_scored" \
    endpoint:endpoint_cut125 peak:peak_cut125
