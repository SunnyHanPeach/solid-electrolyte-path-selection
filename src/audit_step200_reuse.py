"""Audit the historical added-weights/added-Adam replay without reusing its output."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from audit_project_data import write_json
from audit_signed_barriers import SEEDS, file_sha, sha
from run_adam_cross import STEPS
from step200_protocol import STARTS_SHA256


def audit(args):
    protocol = json.loads(args.protocol.read_bytes())
    historical = json.loads(args.historical_manifest.read_bytes())
    if (file_sha(args.states) != STARTS_SHA256 or historical["state_archive_sha256"] != STARTS_SHA256
            or historical["bundle_sha256"] != protocol["bundle_sha256"]):
        raise ValueError("Historical input archives differ")
    result = []
    with ZipFile(args.states) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for seed in SEEDS:
            known = next(row for row in manifest["start_states"] if row["seed"] == seed)
            recorded = next(row for row in protocol["summaries"] if row["seed"] == seed)
            folder = args.historical_runs / f"seed{seed}" / "added-added"
            done = json.loads((folder / "done.json").read_bytes())
            identity = done["identity"]
            if identity != dict(seed=seed, weights="added", adam="added", bundle=protocol["bundle_sha256"],
                                states=STARTS_SHA256, schedule=recorded["replay_order_sha256"],
                                runtime=historical["hardware"], smoke=False,
                                source_script_sha256=historical["source_script_sha256"]):
                raise ValueError(f"Historical run identity differs for seed {seed}")
            if tuple(row["step"] for row in done["training"]) != tuple(range(201, 401)):
                raise ValueError(f"Historical training range differs for seed {seed}")
            for step in STEPS:
                if file_sha(folder / f"predictions_{step}.csv") != done["predictions"][str(step)]:
                    raise ValueError(f"Historical prediction changed for seed {seed}, step {step}")
            added_raw = archive.read(f"seed{seed}/added.pt")
            replay_raw = archive.read(f"seed{seed}/replay.pt")
            if (sha(added_raw), sha(replay_raw)) != (known["augmentation_hash"], known["replay_hash"]):
                raise ValueError(f"Historical common state changed for seed {seed}")
            added = torch.load(io.BytesIO(added_raw), map_location="cpu", weights_only=True)
            replay = torch.load(io.BytesIO(replay_raw), map_location="cpu", weights_only=True)
            adam = added["adam"]["state"]
            if (len(adam) != known["adam_parameter_states"] or len(added["model"]) != known["model_tensors"]
                    or any(s["step"].item() != 200 or s["exp_avg"].shape != s["exp_avg_sq"].shape
                           for s in adam.values())):
                raise ValueError(f"Historical Adam state differs for seed {seed}")
            result.append(dict(seed=seed, added_state_sha256=sha(added_raw), replay_state_sha256=sha(replay_raw),
                               adam_step=200, adam_parameter_states=len(adam), model_tensors=len(added["model"]),
                               replay_order_sha256=identity["schedule"], prediction_hashes=done["predictions"],
                               torch_cpu_rng_sha256=sha(replay["cpu_rng"].numpy().tobytes()),
                               torch_cuda_rng_sha256=sha(replay["cuda_rng"].numpy().tobytes())))
    write_json(args.output, dict(
        reuse_approved=False, planned_continuations=8, historical_runtime=historical["hardware"],
        historical_script_sha256=historical["source_script_sha256"], verified=result,
        reuse_blockers=["Historical Python and NumPy RNG states were not archived",
                        "Exact runtime parity with the new GPU execution is not verified"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("states", "protocol", "historical_manifest", "historical_runs", "output"):
        parser.add_argument(name, type=Path)
    audit(parser.parse_args())
