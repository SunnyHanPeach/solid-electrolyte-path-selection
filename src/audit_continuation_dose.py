"""Verify the matched force coefficients for a Step-200 branch experiment."""
import argparse
import json
from pathlib import Path

from audit_project_data import write_csv, write_json
from audit_signed_barriers import SEEDS, file_sha, rows, sha
from run_force_weighting import force_weights, frozen_schedules, load_bundle


def main(bundle, output):
    _, files = load_bundle(bundle)
    frames = rows(files["training_frames.csv"])
    records, summary = [], []
    for seed in SEEDS:
        replay, appended, added = frozen_schedules(frames, files, seed)
        late_replay, late_appended = replay[200:400], appended[200:400]
        _, after = force_weights(frames, late_appended, added, strength=.75)
        _, full = force_weights(frames, appended, added, strength=.75)
        n_changed = sum(r != a for r, a in zip(late_replay, late_appended))
        for edge, row in after.items():
            records.append(dict(seed=seed, edge_id=edge, full_run_uniform_multiplier=full[edge]["uniform_multiplier"],
                                continuation_uniform_multiplier=row["uniform_multiplier"],
                                continuation_force_dose_unweighted=row["force_coefficient_before"],
                                continuation_force_dose_matched=row["force_coefficient_after"]))
        summary.append(dict(seed=seed, distinct_insertion_batches=n_changed,
                            represented_paths=len(after),
                            replay_order_sha256=sha(json.dumps(late_replay).encode()),
                            augmented_order_sha256=sha(json.dumps(late_appended).encode()),
                            unweighted_interior_dose=sum(r["force_coefficient_before"] for r in after.values()),
                            reduced_interior_dose=sum(r["force_coefficient_after"] for r in after.values())))
    write_csv(output / "continuation_dose.csv", records)
    write_json(output / "continuation_protocol_inputs.json", dict(
        bundle_sha256=file_sha(bundle), start_step=200, final_step=400, peak_discount=.75,
        groups=["replay", "full", "energy_weighted", "matched_uniform"], summaries=summary,
        dose_definition="sum of exposure times atom count divided by total batch atoms, times the force multiplier",
        training_started=False,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.bundle, args.output)
