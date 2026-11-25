"""Locate signed drift on the relative-energy profile without retraining."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_signed_barriers import ARMS, PANELS, SEEDS, STEPS, trajectory_profiles


def metrics(replay, augmented, reference):
    moving_peak = int(np.argmax(replay))
    actual_peak = int(np.argmax(augmented))
    ref_peak = int(np.argmax(reference))
    interior = sorted(range(1, len(reference) - 1), key=lambda i: (-reference[i], i))
    n_high = int(np.ceil(len(interior) / 4))
    shift = augmented - replay
    error_delta = abs(augmented - reference) - abs(replay - reference)
    observed = augmented[actual_peak] - replay[moving_peak]
    fixed_at_replay_peak = shift[moving_peak]
    peak_selection = augmented[actual_peak] - augmented[moving_peak]
    if peak_selection < 0 or not np.isclose(observed, fixed_at_replay_peak + peak_selection,
                                            atol=1e-8, rtol=0):
        raise ValueError("Peak-height decomposition did not close")
    result = dict(
        observed_barrier_shift_meV=float(observed),
        shift_at_replay_peak_meV=float(fixed_at_replay_peak),
        peak_reassignment_uplift_meV=float(peak_selection),
        peak_changed=float(moving_peak != actual_peak),
        endpoint_shift_meV=float(shift[-1]),
        endpoint_abs_error_delta_meV=float(error_delta[-1]),
        shift_at_dft_peak_meV=float(shift[ref_peak]),
    )
    for label, selected in (("high_interior", interior[:n_high]), ("remaining_interior", interior[n_high:])):
        if selected:
            result[f"{label}_shift_meV"] = float(shift[selected].mean())
            result[f"{label}_abs_error_delta_meV"] = float(error_delta[selected].mean())
    return result


def main(args):
    cells = defaultdict(dict)
    for step, seed, profiles, _, references in trajectory_profiles(args.bundle, args.predictions, args.project):
        for key, ref in references.items():
            y, a = (profiles[arm][key] for arm in ARMS)
            cells[step, key[0], key[1]][seed] = metrics(1000 * y, 1000 * a, 1000 * ref)
    result = []
    rng = np.random.default_rng(20261125)
    for step in STEPS:
        for panel in PANELS:
            subset = [data for (t, p, _), data in cells.items() if (t, p) == (step, panel)]
            if len(subset) != (91 if panel == PANELS[0] else 88):
                raise ValueError(f"Incomplete panel: {step}/{panel}")
            assert all(set(x) == set(SEEDS) for x in subset)
            names = {k for x in subset for m in x.values() for k in m}
            metrics_out = {}
            for key in sorted(names):
                points = np.array([[m[s][key] for s in SEEDS] for m in subset
                                   if all(key in m[s] for s in SEEDS)])
                n = len(points)
                values = points.mean(axis=1)
                draw = rng.integers(n, size=(10000, n))
                metrics_out[key] = dict(materials=n, mean=float(values.mean()),
                                        seed_means=points.mean(axis=0).tolist(),
                                        material_interval95=np.quantile(values[draw].mean(axis=1),
                                                                         [.025, .975]).tolist())
            result.append(dict(step=step, panel=panel, metrics=metrics_out))
            if step in (200, 400, 807):
                print(step, panel, {k: round(metrics_out[k]["mean"], 3) for k in (
                    "observed_barrier_shift_meV", "shift_at_replay_peak_meV",
                    "peak_reassignment_uplift_meV", "high_interior_shift_meV",
                    "remaining_interior_shift_meV", "endpoint_shift_meV",
                    "high_interior_abs_error_delta_meV", "remaining_interior_abs_error_delta_meV",
                )})
    protocol = dict(
        high_interior="Per path, the highest DFT relative-energy quarter of interior images, rounded up. "
                      "Ties use the lower image index.",
        fixed_peak_identity="The augmented barrier change equals the relative-energy change at the replay "
                            "peak plus a nonnegative increase from letting the augmented model choose another peak.",
        aggregation="Average between seeds per material, then between materials. Missing interior regions are "
                    "omitted only for the corresponding metric.",
        results=result,
    )
    args.output.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "predictions", "project", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
