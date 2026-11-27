"""Frozen inputs and decisions for the four continuations."""
import json
from math import isclose

from audit_signed_barriers import file_sha, sha
from run_force_weighting import force_weights, frozen_schedules


BRANCHES = ("replay", "full", "energy_weighted", "matched_uniform")
STARTS_SHA256 = "6185d79bc1a4c1dd5581ea2715d3032189815d9ac802b8368d02ceed35897b27"
THRESHOLD_MEV = 0.25


def branch_design(protocol, bundle, states, frames, files, seed):
    """Verify exact schedules and reconstruct per-path force multipliers."""
    spec = json.loads(protocol.read_bytes())
    if (file_sha(states) != STARTS_SHA256 or file_sha(bundle) != spec["bundle_sha256"]
            or (spec["start_step"], spec["final_step"], spec["peak_discount"]) != (200, 400, .75)
            or tuple(spec["groups"]) != BRANCHES):
        raise ValueError("Unexpected frozen branch inputs")
    recorded = next(row for row in spec["summaries"] if row["seed"] == seed)
    replay, augmented, added = frozen_schedules(frames, files, seed)
    replay, augmented = replay[200:400], augmented[200:400]
    replay_sha, augmented_sha = [sha(json.dumps(order).encode()) for order in (replay, augmented)]
    changes = [i for i, (a, b) in enumerate(zip(replay, augmented)) if a != b]
    if (replay_sha, augmented_sha, len(changes)) != (
            recorded["replay_order_sha256"], recorded["augmented_order_sha256"], 24):
        raise ValueError("Continuation batch schedule changed")
    force, details = force_weights(frames, augmented, added, spec["peak_discount"])
    if len(details) != recorded["represented_paths"] or len(details) != 8:
        raise ValueError("Expected exposure to each of the eight paths")
    for field, dose_field in (("unweighted_interior_dose", "force_coefficient_before"),
                              ("reduced_interior_dose", "force_coefficient_after")):
        if not isclose(recorded[field], sum(row[dose_field] for row in details.values()), abs_tol=1e-12):
            raise ValueError(f"Unexpected applied dose: {field}")
    weights = {"replay": {}, "full": {}, "energy_weighted": force["energy_weighted"],
               "matched_uniform": force["uniform_weighted"]}
    smoke = next(i for i in changes if any(
        weights["energy_weighted"].get(j, 1.) != weights["matched_uniform"].get(j, 1.)
        for j in augmented[i]))
    return replay, augmented, weights, smoke


def classify(lo, hi, seed_means):
    """Use an interval over paired materials and an explicit per-seed check."""
    threshold = THRESHOLD_MEV
    if hi < -threshold and all(v < 0 for v in seed_means):
        return "lower"
    if lo > threshold and all(v > 0 for v in seed_means):
        return "higher"
    if lo >= -threshold and hi <= threshold and all(abs(v) <= threshold for v in seed_means):
        return "practically_equivalent"
    return "inconclusive"
