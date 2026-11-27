"""Separate barrier drift, baseline bias, and crossing the DFT reference."""
import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_project_data import write_csv, write_json
from audit_signed_barriers import PANELS, SEEDS, file_sha, load_frozen_data, rows


BIAS_BINS = ("under", "within_25_meV", "over")
BARRIER_BINS = ("lt_250", "250_to_500", "500_to_1000", "ge_1000")
FIELDS = ("start_signed", "replay_signed", "barrier_shift", "absolute_error_change",
          "no_crossing_term", "crossing_correction")


def audit(args):
    _, _, truth = load_frozen_data(args.bundle, args.project)
    with ZipFile(args.bundle) as z:
        index = {(r["panel"], r["edge_id"]): r for r in rows(z.read("evaluation_panels.csv"))}
    if set(index) != set(truth) or len({r["material_id"] for r in index.values()}) != len(index):
        raise ValueError("Expected independent materials with one path per panel")
    path_metrics = rows(args.metrics.read_bytes())
    selected = defaultdict(dict)
    for row in path_metrics:
        step, arm = int(row["step"]), row["arm"]
        if step in (200, 400) and arm in ("replay", "full"):
            key = (row["panel"], row["edge_id"])
            seed = int(row["seed"])
            cell = (step, arm)
            if index[key]["material_id"] != row["material_id"] or cell in selected[key, seed]:
                raise ValueError("Duplicated or mismatched material identity")
            selected[key, seed][cell] = float(row["barrier_signed_error"])
    if set(selected) != {(key, seed) for key in index for seed in SEEDS}:
        raise ValueError("Incomplete paired predictions")
    data, material_rows = {}, []
    for key, meta in sorted(index.items()):
        base = np.asarray([selected[key, seed][200, "replay"] for seed in SEEDS])
        base_full = [selected[key, seed][200, "full"] for seed in SEEDS]
        if not np.array_equal(base, base_full):
            raise ValueError("Common start differs")
        replay = np.asarray([selected[key, seed][400, "replay"] for seed in SEEDS])
        full = np.asarray([selected[key, seed][400, "full"] for seed in SEEDS])
        change = full - replay
        delta = np.abs(full) - np.abs(replay)
        sign = np.where(replay >= 0, 1, -1)
        same_side = sign * change
        correction = 2 * np.maximum(-sign * full, 0)
        np.testing.assert_allclose(delta, same_side + correction, rtol=0, atol=1e-10)
        ref = 1000 * float(truth[key].max())
        bias = BIAS_BINS[np.searchsorted([-25, 25], base.mean(), side="right")]
        band = BARRIER_BINS[np.searchsorted([250, 500, 1000], ref, side="right")]
        elements = set(meta["chemsys"].split("-"))
        composition = "P_and_O" if {"P", "O"}.issubset(elements) else "other"
        values = np.column_stack([base, replay, change, delta, same_side, correction])
        data[key] = {"values": values, "bias": bias, "barrier_band": band,
                     "composition": composition, "chemsys": meta["chemsys"]}
        for seed, vector, r, new in zip(SEEDS, values, replay, full):
            material_rows.append(dict(panel=key[0], edge_id=key[1], material_id=meta["material_id"],
                                      seed=seed, dft_barrier_meV=ref, step200_bias=bias, barrier_band=band,
                                      composition=composition, chemsys=meta["chemsys"],
                                      full_signed_meV=float(new), crosses_reference=bool(r*new < 0),
                                      **dict(zip(FIELDS, map(float, vector)))))
    rng = np.random.default_rng(20261127)
    summary, panel_bins = [], {}
    for panel in PANELS:
        items = [row for (p, _), row in data.items() if p == panel]
        for grouping, groups in (("all", ("all",)), ("bias", BIAS_BINS), ("barrier_band", BARRIER_BINS),
                                 ("composition", ("P_and_O", "other")),
                                 ("chemsys", sorted({r["chemsys"] for r in items}))):
            for group in groups:
                subset = [r["values"] for r in items if grouping == "all" or r[grouping] == group]
                if not subset:
                    continue
                samples = np.asarray(subset)
                n = len(samples)
                seed_avg = samples.mean(axis=1)
                m = seed_avg.mean(axis=0)
                lo, hi = (None, None)
                if n >= 5:
                    draws = rng.integers(n, size=(10_000, n))
                    lo, hi = np.quantile(seed_avg[draws, 3].mean(axis=1), [.025, .975])
                summary.append(dict(panel=panel, grouping=grouping, group=group, materials=n, fraction=n/len(items),
                                    **dict(zip(FIELDS, map(float, m))), lo95=lo, hi95=hi,
                                    negative_shift_fraction=float(np.mean(samples[:, :, 2] < 0)),
                                    crossing_fraction=float(np.mean(samples[:, :, 5] > 0)),
                                    seed17=float(samples[:, 0, 3].mean()), seed29=float(samples[:, 1, 3].mean())))
        panel_bins[panel] = [r for r in summary if r["panel"] == panel and r["grouping"] == "bias"]
    main, aux = [[(r["fraction"], r["absolute_error_change"]) for r in panel_bins[p]] for p in PANELS]
    if len(main) != len(aux) or len(main) != len(BIAS_BINS):
        raise ValueError("Panels do not share all pre-intervention bias bins")
    a, b = np.asarray(main), np.asarray(aux)
    composition = (a[:, 0]-b[:, 0])*(a[:, 1]+b[:, 1])/2
    within = (a[:, 0]+b[:, 0])*(a[:, 1]-b[:, 1])/2
    gap = float(a[:, 0] @ a[:, 1] - b[:, 0] @ b[:, 1])
    np.testing.assert_allclose(gap, composition.sum()+within.sum(), rtol=0, atol=1e-10)
    args.output.mkdir()
    write_csv(args.output / "material_signed_effects.csv", material_rows)
    write_csv(args.output / "strata.csv", summary)
    write_json(args.output / "gap_decomposition.json", dict(
        gap_meV=gap, composition_meV=float(composition.sum()), within_bins_meV=float(within.sum()),
        by_bias=[dict(group=group, composition_meV=float(c), within_bin_meV=float(w))
                 for group, c, w in zip(BIAS_BINS, composition, within)],
        rule="Symmetric standardization on step-200 mean signed-error bins, <−25, [−25,+25), ≥+25 meV",
        bootstrap="10,000 resamples of materials with seeds held fixed; group intervals for n>=5",
        bundle_sha256=file_sha(args.bundle), metrics_sha256=file_sha(args.metrics), script_sha256=file_sha(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "project", "metrics", "output"):
        parser.add_argument(name, type=Path)
    audit(parser.parse_args())
