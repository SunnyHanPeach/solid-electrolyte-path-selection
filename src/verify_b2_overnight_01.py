"""Independently verify the B2 compact archive without model execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import file_sha, rows

SEEDS = (17, 29, 43)


def verify(folder, archive, matrix, checksum):
    actual = file_sha(archive)
    if actual != checksum.read_text(encoding="utf-8").split()[0]:
        raise ValueError("Transport checksum mismatch")
    with ZipFile(matrix) as z:
        locked = json.loads(z.read("plan.json"))
        orders = json.loads(z.read("orders.json"))
        historical = json.loads(z.read("baselines.json"))
        reference = {r["edge_id"]: r for r in rows(z.read("reference_barriers.csv"))}
    matrix_hash = file_sha(matrix)
    with tarfile.open(archive, "r:gz") as z:
        members = {m.name: m for m in z if m.isfile()}
        if any(".." in Path(n).parts or n.startswith("/") for n in members):
            raise ValueError("Unsafe compact member")

        def read(name):
            return z.extractfile(members[name]).read()

        status = json.loads(read("status.json"))
        manifest = json.loads(read("inference_manifest.json"))
        assert status["state"] == "complete" and manifest["matrix_sha256"] == matrix_hash
        slots = status["completed_paths"]
        assert slots == list(range(1, 1 + len(slots))) and status["complete_fits"] == len(slots) * 6
        scores = {(r["arm"], r["seed"]): r for r in manifest["records"]}
        variants = {"pure", *(f"{p}{i}" for p in ("path", "replay") for i in slots)}
        assert set(scores) == {(p, s) for p in variants for s in SEEDS}
        predictions = {}
        fit_count = 0
        for (arm, seed), record in scores.items():
            if arm == "pure":
                assert record["checkpoint_sha256"] == historical[str(seed)]["files"]["model_710.pt"]
            else:
                root = f"fits/{arm}/seed{seed}"
                result = json.loads(read(f"{root}/done.json"))
                schedule = orders[str(seed)][arm]
                assert result["schedule_sha256"] == hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
                assert result["matrix_sha256"] == matrix_hash
                assert result["arm"] == arm and result["seed"] == seed and result["steps"] == len(schedule)
                assert result["files"]["model.pt"] == record["checkpoint_sha256"]
                trace = read(f"{root}/training.csv")
                assert hashlib.sha256(trace).hexdigest() == result["files"]["training.csv"]
                assert [int(r["step"]) for r in rows(trace)] == list(range(1, len(schedule) + 1))
                fit_count += 1
            raw = read(f"predictions/{record['prediction_file']}")
            assert hashlib.sha256(raw).hexdigest() == record["prediction_sha256"]
            profiles = defaultdict(list)
            for r in rows(raw):
                assert r["material_id"] == reference[r["edge_id"]]["material_id"]
                profiles[r["edge_id"]].append((int(r["frame"]), float(r["energy_eV"])))
            assert profiles.keys() == reference.keys()
            for edge, curve in profiles.items():
                curve.sort()
                assert [i for i, _ in curve] == list(range(1, int(reference[edge]["images"]) + 1))
                predicted = 1000 * (max(e for _, e in curve) - curve[0][1])
                predictions[arm, seed, edge] = predicted

        gains = defaultdict(dict)
        published = rows(read("material_gains.csv"))
        required_keys = {(slot, seed, edge) for slot in slots for seed in SEEDS for edge in reference}
        assert len(published) == len(required_keys)
        observed = set()
        max_delta = 0.
        for r in published:
            edge, slot, seed = r["evaluation_edge"], int(r["slot"]), int(r["seed"])
            observed.add((slot, seed, edge))
            assert r["candidate_edge"] == locked["candidates"][slot - 1]["edge_id"]
            assert r["material_id"] == reference[edge]["material_id"]
            true = float(reference[edge]["dft_forward_barrier_meV"])
            errors = {}
            for side, arm in (("pure", "pure"), ("path", f"path{slot}"), ("replay", f"replay{slot}")):
                p = predictions[arm, seed, edge]
                errors[side] = abs(p - true)
                np.testing.assert_allclose([float(r[f"{side}_prediction_meV"]), float(r[f"{side}_error_meV"])],
                                           [p, errors[side]], rtol=0, atol=1e-9)
            for metric, control in (("absolute", "pure"), ("net", "replay")):
                gain = errors[control] - errors["path"]
                max_delta = max(max_delta, abs(gain - float(r[f"{metric}_gain_meV"])))
                np.testing.assert_allclose(gain, float(r[f"{metric}_gain_meV"]), rtol=0, atol=1e-9)
                gains[slot, metric][seed, edge] = gain
        assert observed == required_keys
        intervals = rows(read("intervals.csv"))
        assert len(intervals) == 2 * len(slots)
        # Use per-chemistry sums and material counts to reproduce the interval.
        # This does not build the material-by-resample weight matrix used online.
        chemsystems = sorted({r["chemsys"] for r in reference.values()})
        counts = Counter(r["chemsys"] for r in reference.values())
        n = len(chemsystems)
        resample = np.random.default_rng(20261126).multinomial(n, np.full(n, 1/n), size=10000)
        denominator = resample @ [counts[c] for c in chemsystems]
        for r in intervals:
            sample = gains[int(r["slot"]), r["metric"]]
            cluster_sum = defaultdict(list)
            for edge, truth in reference.items():
                cluster_sum[truth["chemsys"]].append(statistics.mean(sample[s, edge] for s in SEEDS))
            values = resample @ [math.fsum(cluster_sum[c]) for c in chemsystems] / denominator
            lo, hi = np.percentile(values, [2.5, 97.5])
            means = [statistics.mean(sample[s, e] for e in reference) for s in SEEDS]
            np.testing.assert_allclose([float(r[k]) for k in
                    ("low95_meV", "high95_meV", "mean_meV", "seed17_meV", "seed29_meV", "seed43_meV")],
                    [lo, hi, statistics.mean(means), *means], rtol=0, atol=1e-9)
            state = ("positive" if lo > .25 and min(means) > 0 else
                     "negative" if hi < -.25 and max(means) < 0 else
                     "near_zero" if lo >= -.25 and hi <= .25 else "unresolved")
            assert r["status"] == state

        selected = {"inference_manifest.json", "intervals.csv", "path_results.csv", "material_gains.csv",
                    "checkpoint_hashes.csv", "status.json", "training_time.json", "timing_smoke.json"}
        for n in sorted(members):
            if n in selected or (n.startswith(("fits/", "smoke/")) and n.endswith((".log", ".csv", ".json"))):
                target = folder/n
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f:
                    f.write(read(n))
    result = dict(status="pass", compact_sha256=actual, matrix_sha256=matrix_hash, complete_fits=fit_count,
                  materials=len(reference), groups=len(chemsystems), material_rows=len(published),
                  paired_gain_max_difference_meV=max_delta,
                  independence="Recomputed barriers and gains from raw prediction CSVs and intervals from chemistry-group sums.",
                  checkpoint_note="Model bytes were checked on GPU host; this compact archive retains verified checkpoint hashes and traces.")
    with (folder/"verification.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("folder", "archive", "matrix", "checksum"):
        p.add_argument(arg, type=Path)
    verify(**vars(p.parse_args()))
