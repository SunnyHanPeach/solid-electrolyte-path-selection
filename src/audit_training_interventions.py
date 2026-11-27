"""Check intervention inputs without executing an optimizer update."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
from tarfile import open as open_tar

import torch

from audit_project_data import write_csv, write_json
from audit_signed_barriers import ARMS, SEEDS, file_sha, rows, sha
from run_force_weighting import frozen_schedules, load_bundle, omit_frames


def verify_adam(source, bundle_hash, seed, arm, expected, schedule):
    member = f"results/seed{seed}/{arm}/step0200"
    raw = source.extractfile(member + "/checkpoint.pt").read()
    marker = json.loads(source.extractfile(member + "/result.json").read())
    if (marker["seed"], marker["arm"], marker["step"], marker["bundle_sha256"],
        marker["schedule_sha256"], marker["files"]["checkpoint.pt"], sha(raw)) != (
            seed, arm, 200, bundle_hash, sha(json.dumps(schedule).encode()), expected, expected):
        raise ValueError("Invalid Adam intervention starting point")
    state = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    weights = state["model"]
    adam = state["adam"]
    if not adam["state"] or not state["cpu_rng"].numel() or not state["cuda_rng"].numel():
        raise ValueError("Missing optimizer or RNG state")
    layout = {}
    for index, entry in adam["state"].items():
        m, v = entry["exp_avg"], entry["exp_avg_sq"]
        if (entry["step"].item() != 200 or m.shape != v.shape or not torch.isfinite(m).all()
                or not torch.isfinite(v).all() or torch.min(v) < 0):
            raise ValueError("Unexpected Adam tensor layout, step or contents")
        layout[index] = (tuple(m.shape), str(m.dtype))
    return dict(weight_layout={k: (tuple(v.shape), str(v.dtype)) for k, v in weights.items()},
                moment_layout=layout, groups=adam["param_groups"], hash=expected)


def main(args):
    torch.set_num_threads(2)
    _, files = load_bundle(args.bundle)
    frames = rows(files["training_frames.csv"])
    frozen = json.loads(args.verification.read_bytes())
    hashes = {(r["seed"], r["arm"]): r["sha256"] for r in frozen["records"] if r["step"] == 200}
    identity = file_sha(args.bundle)
    states, omissions = [], []
    with open_tar(args.archive, "r:") as source:
        for seed in SEEDS:
            replay, augmented, added = frozen_schedules(frames, files, seed)
            a, b = (verify_adam(source, identity, seed, arm, hashes[seed, arm], schedule)
                    for arm, schedule in zip(ARMS, (replay, augmented)))
            for field in ("weight_layout", "moment_layout", "groups"):
                if a[field] != b[field]:
                    raise ValueError(f"Checkpoint layout differs for {seed}/{field}")
            states.append(dict(seed=seed, start_step=200, model_tensors=len(a["weight_layout"]),
                               adam_parameter_states=len(a["moment_layout"]), replay_hash=a["hash"],
                               augmentation_hash=b["hash"], replay_steps_201_400_sha256=sha(json.dumps(replay[200:400]).encode())))
            schedules = {kind: omit_frames(frames, files, seed, kind) for kind in ("high", "other")}
            for kind, (edited, _, manifest) in schedules.items():
                for step, (original, changed) in enumerate(zip(augmented, edited)):
                    if [frames[i]["n_atoms"] for i in original] != [frames[i]["n_atoms"] for i in changed]:
                        raise ValueError("Omission changed slot or batch atom counts")
                    if replay[step] == original and original != changed:
                        raise ValueError("Omission changed a frozen D0 batch")
                for edge, record in manifest.items():
                    paired = schedules["other" if kind == "high" else "high"][2][edge]
                    if record["presentations"] != paired["presentations"]:
                        raise ValueError("Cross-arm image presentation counts differ")
                    for step, slot, old, replacement in record["replacements"]:
                        if old not in added or replacement != replay[step - 1][slot]:
                            raise ValueError("Replacement is not matched replay")
                        omissions.append(dict(seed=seed, omission_region=kind, source_edge=edge,
                                              step=step, slot=slot, omitted_frame=old, d0_replacement_frame=replacement,
                                              omitted_atom_count=frames[old]["n_atoms"]))
    write_json(args.output / "intervention_preflight.json", dict(
        start_states=states, optimizer_updates_executed=0, bundle_sha256=identity,
        matched_insertions={f"{seed}:{region}": count for (seed, region), count in
                            sorted(Counter((r["seed"], r["omission_region"]) for r in omissions).items())},
    ))
    write_csv(args.output / "omission_preflight_schedule.csv", omissions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "archive", "verification", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
