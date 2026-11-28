"""Record qualifying local dense-grid timings until a fixed deadline."""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from xml.etree import ElementTree


def read_result(folder, branch, prediction):
    end = folder / "end.txt"
    if not end.is_file():
        return {"status": "running" if (folder / "start.txt").is_file() else "queued"}
    fields = dict(line.split("=", 1) for line in end.read_text().splitlines() if "=" in line)
    if not {"elapsed_seconds", "end_utc", "solver_exit"} <= fields.keys():
        return {"status": "awaiting_receipts"}
    if fields["solver_exit"] != "0":
        return {"status": "failed", "end_receipt": fields}
    xml = folder / "scratch" / f"{branch}.save" / "data-file-schema.xml"
    log = folder / "pw.log"
    if not xml.is_file() or not log.is_file():
        return {"status": "awaiting_receipts"}
    root = ElementTree.parse(xml).getroot()
    force = root.find("output/forces")
    force_values = [] if force is None else [float(x) for x in force.text.split()]
    if ("JOB DONE." not in log.read_text() or root.find("closed") is None
            or root.findtext("output/convergence_info/scf_conv/convergence_achieved") != "true"
            or force is None or force.get("dims", "").split() != ["3", "35"]
            or len(force_values) != 105 or not all(map(math.isfinite, force_values))):
        return {"status": "failed_scientific_receipt"}
    elapsed = int(fields["elapsed_seconds"])
    baseline = float(prediction["baseline_wall_seconds"])
    forecast = float(prediction["public_predicted_dense_wall_seconds"])
    return dict(
        status="verified", dense_wall_seconds=elapsed, baseline_wall_seconds=baseline,
        observed_ratio=elapsed/baseline, public_predicted_ratio=float(prediction["public_predicted_ratio"]),
        public_predicted_wall_seconds=forecast,
        observed_minus_prediction_pct=100*(elapsed/forecast-1),
        completed_utc=fields["end_utc"], xml_sha256=sha256(xml.read_bytes()).hexdigest(),
        log_sha256=sha256(log.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("case", "predictions", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("deadline", type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.deadline.tzinfo is None:
        parser.error("deadline must contain the timezone")
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        predictions = {row["anchor"]: row for row in csv.DictReader(handle)}
    pair = {"A": "k_dense_a", "middle": "k_dense_middle"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now(timezone.utc)
        branches = {anchor: read_result(args.case / branch, branch, predictions[anchor])
                    for anchor, branch in pair.items()}
        all_verified = all(row["status"] == "verified" for row in branches.values())
        reached_deadline = now >= args.deadline
        pending = args.output.with_suffix(".partial")
        pending.write_text(json.dumps(dict(
            observed_utc=now.isoformat(), deadline=args.deadline.isoformat(),
            all_verified=all_verified, reached_deadline=reached_deadline, branches=branches,
        ), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.replace(args.output)
        if reached_deadline or all_verified:
            break
        time.sleep(min(300, max(0, (args.deadline-now).total_seconds())))


if __name__ == "__main__":
    main()
