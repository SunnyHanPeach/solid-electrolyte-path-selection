"""Score the eight common-start continuations on the frozen historical panels."""
import argparse
from collections import defaultdict
import json
from math import ceil, isclose
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_project_data import write_csv, write_json
from audit_signed_barriers import PANELS, SEEDS, file_sha, load_frozen_data, rows, sha
from run_adam_cross import STEPS
from run_force_weighting import load_bundle
from step200_protocol import BRANCHES, STARTS_SHA256, THRESHOLD_MEV, branch_design, classify


CONTRASTS = (("full", "replay"), ("energy_weighted", "full"), ("matched_uniform", "full"),
             ("energy_weighted", "matched_uniform"), ("energy_weighted", "replay"),
             ("matched_uniform", "replay"))
METRICS = ("barrier_absolute_error", "barrier_signed_error", "fixed_peak_signed_error",
           "high_profile_absolute_error", "old_peak_relative_energy_change", "peak_switch_contribution")


def read_runs(args, frames):
    expected = {(r["panel"], r["edge_id"], r["material_id"], int(r["frame"])) for r in frames}
    source = {}
    runtime, scripts = set(), set()
    _, files = load_bundle(args.bundle)
    training_frames = rows(files["training_frames.csv"])
    with ZipFile(args.states) as z:
        starts = {r["seed"]: r for r in json.loads(z.read("manifest.json"))["start_states"]}
    bundle_hash, protocol_hash = file_sha(args.bundle), file_sha(args.protocol)
    for seed in SEEDS:
        start = starts[seed]
        replay, augmented, weights, _ = branch_design(args.protocol, args.bundle, args.states,
                                                       training_frames, files, seed)
        initial_hashes = set()
        for arm in BRANCHES:
            folder = args.runs / f"seed{seed}" / arm
            done = json.loads((folder / "done.json").read_bytes())
            identity = done["identity"]
            order = replay if arm == "replay" else augmented
            keys = dict(seed=seed, arm=arm, weights="added", adam="added", smoke=False,
                        bundle=bundle_hash, protocol_sha256=protocol_hash, states=STARTS_SHA256,
                        start_state_sha256=start["augmentation_hash"], rng_state_sha256=start["replay_hash"],
                        schedule=sha(json.dumps(order).encode()))
            if any(identity.get(k) != v for k, v in keys.items()):
                raise ValueError(f"Frozen inputs differ for {seed}/{arm}")
            expected_steps = [(step, ids, [weights[arm].get(i, 1.) for i in ids])
                              for step, ids in enumerate(order, 201)]
            actual_steps = [(r["step"], r["frame_indices"], r["force_multipliers"]) for r in done["training"]]
            # Per-path dose totals can differ at floating-point rounding precision across hosts.
            if len(actual_steps) != len(expected_steps) or any(
                    (step, ids) != (ref_step, ref_ids) or len(values) != len(ref_values) or any(
                        not isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(values, ref_values))
                    for (step, ids, values), (ref_step, ref_ids, ref_values) in zip(actual_steps, expected_steps)):
                raise ValueError(f"Executed schedule or force multipliers differ for {seed}/{arm}")
            runtime.add(json.dumps(identity["runtime"], sort_keys=True))
            scripts.add(identity["source_script_sha256"])
            initial_hashes.add(done["predictions"]["200"])
            for step in STEPS:
                raw = (folder / f"predictions_{step}.csv").read_bytes()
                if sha(raw) != done["predictions"][str(step)]:
                    raise ValueError(f"Prediction hash differs for {seed}/{arm}/{step}")
                values = rows(raw)
                found = {(r["panel"], r["edge_id"], r["material_id"], int(r["frame"])) for r in values}
                if len(values) != len(expected) or found != expected:
                    raise ValueError(f"Evaluation frames differ for {seed}/{arm}/{step}")
                grouped = defaultdict(dict)
                for row in values:
                    grouped[row["panel"], row["edge_id"]][int(row["frame"])] = float(row["energy_eV"])
                for key, energies in grouped.items():
                    profile = np.array([energies[i] for i in sorted(energies)])
                    if not np.isfinite(profile).all():
                        raise ValueError(f"Nonfinite energy for {key}")
                    source[step, seed, arm, key] = 1000 * (profile - profile[0])
        if len(initial_hashes) != 1:
            raise ValueError(f"The four arms do not share the seed {seed} start")
    if len(runtime) != 1 or len(scripts) != 1:
        raise ValueError("Continuations used different runtimes or runner versions")
    return source, json.loads(next(iter(runtime))), next(iter(scripts))


def profile_metrics(pred, ref, high):
    return np.array([abs(pred.max() - ref.max()), pred.max() - ref.max(),
                     pred[high[0]] - ref[high[0]], np.abs(pred[high] - ref[high]).mean()])


def peak_decomposition(new, old):
    p, q = int(old.argmax()), int(new.argmax())
    drift, switch = new[p] - old[p], new[q] - new[p]
    np.testing.assert_allclose(drift + switch, new.max() - old.max(), atol=1e-9, rtol=0)
    return drift, switch


def score(args):
    frames, _, truth = load_frozen_data(args.bundle, args.project)
    source, runtime, script = read_runs(args, frames)
    material = {(r["panel"], r["edge_id"]): r["material_id"] for r in frames}
    rng = np.random.default_rng(20261127)
    arm_rows, contrast_rows, summary = [], [], []
    for panel, count in zip(PANELS, (91, 88)):
        keys = sorted(key for key in truth if key[0] == panel)
        if len(keys) != count or len({material[key] for key in keys}) != count:
            raise ValueError("Expected one path per frozen material")
        draws = rng.integers(count, size=(10_000, count))
        ref = {key: 1000 * truth[key] for key in keys}
        high = {key: sorted(range(1, len(curve)-1), key=lambda i: (-curve[i], i))[:ceil((len(curve)-2)/4)]
                for key, curve in ref.items()}
        for step in STEPS:
            effects = defaultdict(list)
            for seed in SEEDS:
                for key in keys:
                    label = dict(step=step, panel=panel, seed=seed, edge_id=key[1], material_id=material[key])
                    curves = {arm: source[step, seed, arm, key] for arm in BRANCHES}
                    metrics = {arm: profile_metrics(curve, ref[key], high[key]) for arm, curve in curves.items()}
                    for arm, vals in metrics.items():
                        arm_rows.append(dict(**label, arm=arm, dft_peak_index=high[key][0],
                                             model_peak_index=int(curves[arm].argmax()),
                                             **dict(zip(METRICS[:4], map(float, vals)))))
                    for left, right in CONTRASTS:
                        name = f"{left}_minus_{right}"
                        vals = np.r_[metrics[left]-metrics[right], peak_decomposition(curves[left], curves[right])]
                        effects[name].append(vals)
                        contrast_rows.append(dict(**label, contrast=name, **dict(zip(METRICS, map(float, vals)))))
            for name, values in effects.items():
                data = np.asarray(values).reshape(len(SEEDS), count, len(METRICS))
                seed_mean = data.mean(axis=1)
                averaged = data.mean(axis=0)
                low, high_ci = np.quantile(averaged[draws].mean(axis=1), [.025, .975], axis=0)
                for i, metric in enumerate(METRICS):
                    decision = "diagnostic"
                    if step == 400 and panel == "primary91" and metric == METRICS[0]:
                        decision = classify(low[i], high_ci[i], seed_mean[:, i])
                    summary.append(dict(step=step, panel=panel, contrast=name, metric=metric, materials=count,
                                        mean_meV=float(averaged[:, i].mean()), lo95_meV=float(low[i]),
                                        hi95_meV=float(high_ci[i]), seed17_meV=float(seed_mean[0, i]),
                                        seed29_meV=float(seed_mean[1, i]), decision=decision))
    args.output.mkdir()
    write_csv(args.output / "material_metrics.csv", arm_rows)
    write_csv(args.output / "material_contrasts.csv", contrast_rows)
    write_csv(args.output / "contrasts.csv", summary)
    write_json(args.output / "manifest.json", dict(
        runtime=runtime, runner_sha256=script, source_bundle_sha256=file_sha(args.bundle),
        protocol_sha256=file_sha(args.protocol), scorer_sha256=file_sha(Path(__file__)),
        seeds=SEEDS, milestones=STEPS, threshold_meV=THRESHOLD_MEV,
        decision_endpoint="primary91 forward barrier MAE at step 400",
        bootstrap="10,000 paired resamples by material, conditional on fixed seeds 17 and 29",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "states", "protocol", "project", "runs", "output"):
        parser.add_argument(name, type=Path)
    score(parser.parse_args())
