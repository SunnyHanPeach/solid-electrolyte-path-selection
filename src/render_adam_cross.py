"""Plot fixed-batch Adam and model-parameter effects on barrier MAE."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_project_data import read_csv
from render_project_audit import BLUE, RUST, TEAL, label, save, style


def plot(path, output):
    data = read_csv(path)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.38))
    fig.subplots_adjust(left=.102, right=.974, bottom=.16, top=.894, wspace=.42, hspace=.71)
    panels = (("primary91", "Main panel"), ("historical_support", "Auxiliary panel"))
    metrics = (("absolute_mae", "Endpoint MAE"), ("change_from_step200", "MAE change"))
    called = ((0, "adam_main", "Adam state", TEAL, "o"),
              (1, "weights_main", "Model parameters", BLUE, "s"),
              (2, "interaction", "Interaction", RUST, "^"))
    for row, (metric, metric_name) in enumerate(metrics):
        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            for offset, effect, name, color, marker in called:
                subset = sorted((r for r in data if (r["panel"], r["metric"], r["effect"]) ==
                                 (panel, metric, effect)), key=lambda r: int(r["step"]))
                if len(subset) != 6:
                    raise ValueError("Expected six frozen trajectory checkpoints")
                x = np.arange(len(subset)) + (offset-1)*.1
                effect_values, lower, upper = np.asarray([[float(r[k]) for k in ("mean_meV", "lo95_meV", "hi95_meV")]
                                                         for r in subset]).T
                ax.errorbar(x, effect_values, yerr=[effect_values-lower, upper-effect_values], marker=marker,
                            color=color, label=name, lw=.8, capsize=2, ms=3.4)
            ax.axhline(0, ls="--", color="#9ba5ac", lw=.8)
            ax.grid(axis="y")
            ax.set_xticks(np.arange(len(subset)), [r["step"] for r in subset])
            ax.set_xlabel("Training update")
            ax.set_ylabel("Factorial effect (meV)")
            label(ax, "abcd"[2*row+col], f"{title} · {metric_name}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.54, .036))
    save(fig, output, "adam_cross_factorial_effects")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("contrasts", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    plot(args.contrasts, args.output)
