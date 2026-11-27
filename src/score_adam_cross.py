"""Score the crossed-Adam continuations against the two frozen historical panels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_project_data import write_csv, write_json
from audit_signed_barriers import SEEDS, file_sha, load_frozen_data, rows, sha
from run_adam_cross import FACTORS, STEPS


def effect_vectors(x):
    rr, ra, ar, aa = x.T
    return {
        "adam_main": ((ra - rr) + (aa - ar)) / 2,
        "weights_main": ((ar - rr) + (aa - ra)) / 2,
        "interaction": (aa - ar) - (ra - rr),
        "both_added_vs_both_replay": aa - rr,
        "adam_with_replay_weights": ra - rr,
        "adam_with_added_weights": aa - ar,
        "weights_with_replay_adam": ar - rr,
        "weights_with_added_adam": aa - ra,
    }


def score(args):
    frames, schedules, truth = load_frozen_data(args.bundle, args.project)
    bundle_hash = file_sha(args.bundle)
    arms = [f"{w}-{a}" for w in FACTORS for a in FACTORS]
    expected = {(row["panel"], row["edge_id"], row["material_id"], int(row["frame"])) for row in frames}
    material = {(panel, edge): mid for panel, edge, mid, _ in expected}
    if len(material) != len({(p, mid) for (p, _), mid in material.items()}):
        raise ValueError("Multiple paths would change the material-level weighting")
    environment = set()
    source_scripts = set()
    state_archives = set()
    paired = defaultdict(dict)
    individual = []
    for seed in SEEDS:
        schedule = sha(json.dumps(schedules[seed]["protected_replay"][200:400]).encode())
        for arm in arms:
            directory = args.runs / f"seed{seed}" / arm
            done = json.loads((directory / "done.json").read_bytes())
            w, a = arm.split("-")
            identity = done["identity"]
            if (identity["seed"], identity["weights"], identity["adam"], identity["bundle"], identity["schedule"],
                    identity["smoke"]) != (seed, w, a, bundle_hash, schedule, False):
                raise ValueError(f"Unexpected experiment identity: {seed}/{arm}")
            if tuple(row["step"] for row in done["training"]) != tuple(range(201, 401)):
                raise ValueError(f"Incomplete updates: {seed}/{arm}")
            environment.add(json.dumps(identity["runtime"], sort_keys=True))
            source_scripts.add(identity["source_script_sha256"])
            state_archives.add(identity["states"])
            for step in STEPS:
                path = directory / f"predictions_{step}.csv"
                raw = path.read_bytes()
                if sha(raw) != done["predictions"][str(step)]:
                    raise ValueError(f"Prediction checksum differs: {path}")
                predictions = rows(raw)
                found = {(r["panel"], r["edge_id"], r["material_id"], int(r["frame"])) for r in predictions}
                if len(predictions) != len(expected) or found != expected:
                    raise ValueError(f"Frozen frame coverage differs: {seed}/{arm}/{step}")
                groups = defaultdict(dict)
                for row in predictions:
                    groups[row["panel"], row["edge_id"]][int(row["frame"])] = float(row["energy_eV"])
                for key, energies in groups.items():
                    profile = np.array([energies[i] for i in sorted(energies)])
                    if not np.isfinite(profile).all():
                        raise ValueError("Nonfinite energy")
                    ref = 1000 * float(truth[key].max())
                    pred = 1000 * float(profile.max() - profile[0])
                    signed = pred - ref
                    paired[step, key[0], seed, key[1]][arm] = abs(signed)
                    individual.append(dict(step=step, panel=key[0], edge_id=key[1], material_id=material[key], seed=seed,
                                           arm=arm, reference_barrier_meV=ref, predicted_barrier_meV=pred,
                                           signed_error_meV=signed, absolute_error_meV=abs(signed)))
    if len(environment) != 1 or len(source_scripts) != 1 or len(state_archives) != 1:
        raise ValueError("The eight arms used different saved-state bundles or runtimes")
    panels = sorted({p for p, _ in truth})
    rng = np.random.default_rng(20261127)
    summary = []
    outcome = []
    for panel in panels:
        edges = sorted(edge for p, edge in truth if p == panel)
        draws = rng.integers(len(edges), size=(10000, len(edges)))
        baseline = np.asarray([[[paired[200, panel, seed, edge][arm] for arm in arms] for edge in edges]
                               for seed in SEEDS])
        if not (np.array_equal(baseline[:, :, 0], baseline[:, :, 1]) and
                np.array_equal(baseline[:, :, 2], baseline[:, :, 3])):
            raise ValueError("Initial predictions differ between identical model parameters")
        for step in STEPS:
            absolute = np.asarray([[[paired[step, panel, seed, edge][arm] for arm in arms] for edge in edges]
                                   for seed in SEEDS])
            for metric, data in (("absolute_mae", absolute), ("change_from_step200", absolute-baseline)):
                for seed, vals in zip(SEEDS, data):
                    outcome.extend(dict(step=step, panel=panel, metric=metric, seed=seed, arm=arm,
                                        mean_meV=float(vals[:, i].mean())) for i, arm in enumerate(arms))
                seed_effects = [effect_vectors(data[i]) for i in range(len(SEEDS))]
                for name in seed_effects[0]:
                    pair = np.stack([item[name] for item in seed_effects])
                    averaged = pair.mean(axis=0)
                    lo, hi = np.quantile(averaged[draws].mean(axis=1), [.025, .975])
                    summary.append(dict(step=step, panel=panel, metric=metric, effect=name, materials=len(edges),
                                        mean_meV=float(averaged.mean()), lo95_meV=float(lo), hi95_meV=float(hi),
                                        seed17_meV=float(pair[0].mean()), seed29_meV=float(pair[1].mean())))
    args.output.mkdir()
    write_csv(args.output/"material_energies.csv", individual)
    write_csv(args.output/"arm_means.csv", outcome)
    write_csv(args.output/"factorial_contrasts.csv", summary)
    write_json(args.output/"manifest.json", dict(bundle_sha256=bundle_hash, frames=len(expected), models=len(arms)*len(SEEDS),
               optimizer_steps=[201, 400], milestones=STEPS, hardware=json.loads(next(iter(environment))),
               bootstrap="paired historical materials, conditional on seeds 17 and 29, 10,000 replicates",
               source_script_sha256=next(iter(source_scripts)), state_archive_sha256=next(iter(state_archives))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("bundle", "project", "runs", "output"):
        parser.add_argument(n, type=Path)
    score(parser.parse_args())
