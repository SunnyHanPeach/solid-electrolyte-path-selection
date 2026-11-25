"""Render the Phase 1 feature coverage and paired gain figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {"absolute": "#306699", "net": "#bf6224"}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def features(args, plan):
    table = load_csv(args.design / "candidate_path_features.csv")
    columns = [c for group in plan["path_feature_groups"] for c in group]
    values = np.asarray([[float(r[c]) for c in columns] for r in table])
    eligible = np.asarray([r["eligible"]=="1" for r in table])
    qlo, qhi = np.quantile(values[eligible], [.25,.75], axis=0)
    scaled = (values-values[eligible].mean(axis=0))/np.where(qhi > qlo, qhi-qlo, 1)
    _, singular, vectors = np.linalg.svd(scaled[eligible], full_matrices=False)
    variance = (singular**2)/(singular**2).sum()
    xy = scaled @ vectors[:2].T
    fig, ax = plt.subplots(figsize=(7,4.4), layout="constrained")
    ax.scatter(*xy[eligible].T,color="#71879a",alpha=.7,s=26,label="Eligible")
    ax.scatter(*xy[~eligible].T,color="#c1ccd3",marker="x",s=32,label="Other candidates")
    lookup = {r["edge_id"]:i for i,r in enumerate(table)}
    for n, edge in enumerate(plan["selected"],1):
        x,y = xy[lookup[edge]]
        ax.scatter(x,y,color="#b0442c",s=48)
        ax.annotate(f"Path {n}",(x,y),xytext=(5,6),textcoords="offset points")
    ax.set(xlabel=f"PC1 ({variance[0]:.0%})",ylabel=f"PC2 ({variance[1]:.0%})",
           title="Pre-DFT feature coverage")
    ax.legend(frameon=False,loc="best")
    fig.savefig(args.output/"feature_coverage.png",dpi=180)
    plt.close(fig)


def gains(args, plan):
    summary = json.loads((args.results/"summary.json").read_bytes())
    material_rows = load_csv(args.results/"paired_materials.csv")
    fig, ax = plt.subplots(figsize=(7,4.2),layout="constrained")
    ax.axvspan(-summary["margin_meV"],summary["margin_meV"],color="#e8ebed")
    ax.axvline(0,color="#56616e",linewidth=1)
    for n, edge in enumerate(plan["selected"],1):
        for metric, shift in (("absolute",-.13),("net",.13)):
            point = summary["results"][edge][metric]
            lo,hi=point["interval95"]
            mean=point["mean"]
            ax.errorbar(mean,n+shift,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],
                        marker="o",capsize=3,color=PALETTE[metric],
                        label=metric.title() if n==1 else None)
            ax.scatter(point["seed_means"],[n+shift]*3,color=PALETTE[metric],alpha=.3,s=13)
    ax.set(yticks=(1,2,3),yticklabels=["Path 1","Path 2","Path 3"],
           xlabel="Barrier MAE gain (meV)",title="Absolute and exposure-adjusted gains")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    fig.savefig(args.output/"paired_gains.png",dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(10,3.6),sharey=True,layout="constrained")
    for n,(ax,edge) in enumerate(zip(axes,plan["selected"]),1):
        for col,metric in enumerate(("absolute","net")):
            by_material = {}
            for row in material_rows:
                if row["path"]==edge and row["metric"]==metric:
                    by_material.setdefault(row["material"],[]).append(float(row["gain_meV"]))
            v = [np.mean(items) for items in by_material.values()]
            jitter = np.random.default_rng(20261125).uniform(-.14,.14,len(v))
            ax.scatter(col+jitter,v,s=10,alpha=.55,color=PALETTE[metric],edgecolors="none")
        ax.axhline(0,color="#56616e",linewidth=1)
        ax.set(xticks=(0,1),xticklabels=("Absolute","Net"),title=f"Path {n}")
    axes[0].set_ylabel("Gain per material (meV)")
    fig.savefig(args.output/"material_gains.png",dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--results",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((args.design/"plan.json").read_bytes())
    features(args,plan)
    if args.results:
        gains(args,plan)


if __name__ == "__main__":
    main()
