"""Audit signed barrier drift and panel differences on frozen trajectory predictions."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from zipfile import ZipFile

import numpy as np

PANELS = ("primary91", "historical_support")
SEEDS = (17, 29)
STEPS = (8, 64, 200, 400, 600, 807)
ARMS = ("protected_replay", "protected8_batch1_random")
REFERENCE_SHA256 = "b7a99d89337902e9e1da319f57547170fdb132bf15bf6ffef03a0140e2207d7f"
PREDICTION_FIELDS = ("panel", "step", "seed", "edge_id", "material_id",
                     "replay_signed_error_meV", "augmented_signed_error_meV")


def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_data(bundle, project):
    with ZipFile(bundle) as z:
        plan = json.loads(z.read("plan.json"))
        manifest = {}
        for name in ("evaluation_panels.csv", "evaluation_frames.csv", "jobs.json"):
            raw = z.read(name)
            if sha(raw) != plan["sha256"][name]:
                raise ValueError(f"Frozen member failed its checksum: {name}")
            manifest[name] = raw
    index = rows(manifest["evaluation_panels.csv"])
    frames = rows(manifest["evaluation_frames.csv"])
    schedules = {entry["seed"]: entry["jobs"] for entry in json.loads(manifest["jobs.json"])}
    sys.path.insert(0, str(project / "experiment2"))
    from prepare_data import metadata, raw_frames

    source = project / "\u8fc1\u79fb\u8def\u5f84\u6570\u636e" / "nebDFT2k.zip"
    if file_sha(source) != REFERENCE_SHA256:
        raise ValueError("DFT archive failed its checksum")
    truth = {}
    with ZipFile(source) as z:
        for r in index:
            edge = r["edge_id"]
            energies = {i: float(metadata(header)[b"energy"])
                        for i, _, _, header, _ in raw_frames(z, f"nebDFT2k/{edge}_relaxed.xyz")}
            if sorted(energies) != list(range(1, int(r["images"]) + 1)):
                raise ValueError(f"Incomplete DFT path: {edge}")
            curve = np.array([energies[i] for i in sorted(energies)])
            truth[r["panel"], edge] = curve - curve[0]
    return frames, schedules, truth


def trajectory_profiles(bundle, predictions, project):
    frames, schedules, truth = load_frozen_data(bundle, project)
    bundle_hash = file_sha(bundle)

    def key(row):
        return row["panel"], row["edge_id"], row["material_id"], int(row["frame"])

    expected = {key(r) for r in frames}
    material_ids = {(panel, edge): material for panel, edge, material, _ in expected}
    with tarfile.open(predictions, "r:*") as z:
        def read(member):
            with z.extractfile(member) as f:
                return f.read()

        for step in STEPS:
            for seed in SEEDS:
                profiles = {}
                for arm in ARMS:
                    path = f"results/seed{seed}/{arm}/step{step:04}"
                    record = json.loads(read(f"{path}/result.json"))
                    order_hash = sha(json.dumps(schedules[seed][arm]).encode())
                    if (record["step"], record["seed"], record["arm"], record["bundle_sha256"],
                        record["schedule_sha256"]) != (step, seed, arm, bundle_hash, order_hash):
                        raise ValueError(f"Frozen experiment identity mismatch: {path}")
                    raw = read(f"{path}/blind_predictions.csv")
                    if sha(raw) != record["files"]["blind_predictions.csv"]:
                        raise ValueError(f"Prediction failed checksum: {path}")
                    received = rows(raw)
                    if {key(row) for row in received} != expected or len(received) != len(expected):
                        raise ValueError(f"Evaluation frames mismatch: {path}")
                    energy = defaultdict(dict)
                    for row in received:
                        panel, edge, _, i = key(row)
                        val = float(row["energy_eV"])
                        if not np.isfinite(val):
                            raise ValueError(f"Nonfinite energy: {path}/{edge}")
                        energy[panel, edge][i] = val
                    profiles[arm] = {
                        k: np.array([values[i] for i in sorted(values)]) - values[1]
                        for k, values in energy.items()
                    }
                yield step, seed, profiles, material_ids, truth


def extract(args):
    # Validate signed derivation against the previously published absolute differences.
    with args.absolute.open(newline="", encoding="utf-8") as f:
        old = {(r["panel"], int(r["step"]), int(r["seed"]), r["edge"]): float(r["delta_meV"])
               for r in csv.DictReader(f)}
    signed = []
    for step, seed, profiles, material_ids, truth in trajectory_profiles(args.bundle, args.predictions, args.project):
        for panel, edge in sorted(truth):
            key = (panel, edge)
            barrier = max(truth[key])
            r, a = (1000 * (max(profiles[arm][key]) - barrier) for arm in ARMS)
            np.testing.assert_allclose(abs(a) - abs(r), old[panel, step, seed, edge], atol=1e-8, rtol=0)
            signed.append([panel, step, seed, edge, material_ids[key], r, a])
    if len(signed) != len(old):
        raise ValueError("Signed coverage differs from the earlier score audit")
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PREDICTION_FIELDS)
        writer.writerows(signed)


def load_matrices(path):
    with path.open(newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))
    arrays = {}
    for panel in PANELS:
        indices = [(r["edge_id"], r["material_id"]) for r in entries if r["panel"] == panel]
        ids = sorted(set(indices))
        if len({m for _, m in ids}) != len(ids):
            raise ValueError("Expected one evaluation path per material")
        records = {(int(r["step"]), int(r["seed"]), r["edge_id"], r["material_id"]):
                   (float(r["replay_signed_error_meV"]), float(r["augmented_signed_error_meV"]))
                   for r in entries if r["panel"] == panel}
        if len(records) != len(indices) or len(records) != len(STEPS) * len(SEEDS) * len(ids):
            raise ValueError(f"Incomplete or repeated signed data: {panel}")
        matrix = np.asarray([[[records[step, seed, edge, m] for edge, m in ids]
                              for seed in SEEDS] for step in STEPS])
        if not np.isfinite(matrix).all():
            raise ValueError(f"Invalid signed data: {panel}")
        arrays[panel] = matrix[..., 0], matrix[..., 1]
    return arrays


def describe(v, sample):
    v = v.mean(axis=0)
    return dict(mean_meV=float(v.mean()), interval95_meV=np.quantile(
        v[sample].mean(axis=1), [.025, .975]).tolist())


def analyze(args):
    matrices = load_matrices(args.signed)
    if [v[0].shape[-1] for v in matrices.values()] != [91, 88]:
        raise ValueError("Unexpected panel counts")
    # Freeze groups using the first saved replay checkpoint, which saw no candidate data.
    baseline_bins = {}
    samples = {}
    rng = np.random.default_rng(20261125)
    for panel, (rep, _) in matrices.items():
        anchor = rep[0].mean(axis=0)
        baseline_bins[panel] = np.digitize(anchor, [-25.0, 25.0])
        n = len(anchor)
        samples[panel] = rng.integers(n, size=(20000, n))
    weights = np.bincount(np.concatenate(list(baseline_bins.values())), minlength=3)
    weights = weights / weights.sum()
    shuffle_rng = np.random.default_rng(20261126)
    steps = []
    for i, step in enumerate(STEPS):
        changes = {k: aug[i] - rep[i] for k, (rep, aug) in matrices.items()}
        shared = np.concatenate(list(changes.values()), axis=1).mean(axis=1, keepdims=True)
        panel_data, contrasts = {}, defaultdict(list)
        for panel, other in zip(PANELS, PANELS[::-1]):
            rep, aug = (v[i] for v in matrices[panel])
            shift = changes[panel]
            actual = abs(aug) - abs(rep)
            shared_only = abs(rep + shared) - abs(rep)
            cross_only = abs(rep + changes[other].mean(axis=1, keepdims=True)) - abs(rep)
            residual = actual - shared_only
            # Keep both seed shifts for a material together while permuting material assignments.
            n_materials = rep.shape[1]
            order = shuffle_rng.permuted(np.broadcast_to(np.arange(n_materials), (5000, n_materials)), axis=1)
            shuffled = (abs(rep[:, None, :] + shift[:, order]) - abs(rep[:, None, :])).mean(axis=(0, 2))
            actual_vs_randomized_assignment = dict(
                shuffled_mean_meV=float(shuffled.mean()),
                shuffled_reference_interval95_meV=np.quantile(shuffled, [.025, .975]).tolist(),
                actual_minus_shuffled_mean_meV=float(actual.mean()-shuffled.mean()),
            )
            by_group = []
            means = []
            for g, name in enumerate(("underpredicted", "near_zero", "overpredicted")):
                selected = baseline_bins[panel] == g
                n = int(selected.sum())
                mean = float(actual[:, selected].mean()) if n else None
                means.append(mean)
                by_group.append(dict(group=name, materials=n, actual_delta_meV=mean,
                                     signed_barrier_shift_meV=float(shift[:, selected].mean()) if n else None,
                                     replay_signed_error_meV=float(rep[:, selected].mean()) if n else None,
                                     augmented_signed_error_meV=float(aug[:, selected].mean()) if n else None,
                                     seed_means_meV=actual[:, selected].mean(axis=1).tolist() if n else []))
            balanced = float(np.dot(means, weights)) if None not in means else None
            subset = samples[panel]
            panel_data[panel] = dict(
                signed_shift=describe(shift, subset), actual=describe(actual, subset),
                pooled_shared_shift_only=describe(shared_only, subset),
                opposite_panel_shift_only=describe(cross_only, subset),
                residual_after_shared_shift=describe(residual, subset),
                randomized_material_assignment=actual_vs_randomized_assignment,
                lower_barriers_materials=int((shift.mean(axis=0) < 0).sum()),
                increased_absolute_error_materials=int((actual.mean(axis=0) > 0).sum()),
                seed_actual_meV=actual.mean(axis=1).tolist(),
                grouped_at_step8_replay=by_group,
                actual_balanced_for_step8_bins_meV=balanced,
            )
            for k, v in (("actual", actual), ("shared_shift_only", shared_only), ("residual", residual)):
                contrasts[k].append(v.mean(axis=0)[subset].mean(axis=1))
        gap = {k: dict(interval95_meV=np.quantile(v[0] - v[1], [.025, .975]).tolist())
               for k, v in contrasts.items()}
        main, support = (panel_data[p] for p in PANELS)
        for k, field in (("actual", "actual"), ("shared_shift_only", "pooled_shared_shift_only"),
                         ("residual", "residual_after_shared_shift")):
            gap[k]["mean_meV"] = main[field]["mean_meV"] - support[field]["mean_meV"]
        reweighted = [p["actual_balanced_for_step8_bins_meV"] for p in (main, support)]
        steps.append(dict(step=step, pooled_uniform_signed_shift_per_seed_meV=shared.ravel().tolist(),
                          panels=panel_data, gap_main_minus_support=gap,
                          gap_after_step8_bias_balance_meV=(reweighted[0] - reweighted[1]
                                                           if None not in reweighted else None)))
    info = dict(
        definition="Signed error is predicted minus reference forward barrier. Positive absolute-error delta is harm.",
        reference_groups="Step 8 replay only, averaged over seeds before defining groups below -25, near zero, above +25 meV.",
        uncertainty="Paired material bootstrap conditional on fixed seeds and estimated shifts. Historical exploratory audit.",
        interpretation=("Counterfactuals apply a uniform shift or permute the observed paired shifts across materials. "
                        "Permutation intervals are descriptive, conditional on the panel, and do not imply causation."),
        steps=steps,
    )
    args.output.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for step in steps:
        if step["step"] < 200:
            continue
        print(step["step"], {p: {k: round(step["panels"][p][k]["mean_meV"], 3)
                                 for k in ("signed_shift", "actual", "pooled_shared_shift_only",
                                           "opposite_panel_shift_only", "residual_after_shared_shift")}
                             for p in PANELS})
        print("main_minus_support", step["gap_main_minus_support"],
              "balanced", step["gap_after_step8_bias_balance_meV"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("extract")
    for name in ("bundle", "predictions", "project", "absolute", "output"):
        exp.add_argument(name, type=Path)
    report = commands.add_parser("analyze")
    for name in ("signed", "output"):
        report.add_argument(name, type=Path)
    opts = parser.parse_args()
    (extract if opts.command == "extract" else analyze)(opts)
