"""Render the predeclared H1-A incremental-value comparison on the cloud host."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase_h1_gpu import METHODS, TARGETS, read


def main():
    script=argparse.ArgumentParser(description=__doc__)
    script.add_argument("gpu_results",type=Path)
    script.add_argument("output",type=Path)
    args=script.parse_args()
    args.output.mkdir()
    stats=read(args.gpu_results/"utility_metrics.csv")
    uncertainty=read(args.gpu_results/"paired_uncertainty.csv")
    estimands=["Absolute gain","Net gain"]
    folds=("LOPO","anion-family-held-out")
    fig,ax=plt.subplots(2,2,figsize=(10.8,6.6),layout="constrained")
    colors=["#657c94","#16769c","#cd7340"]
    for j,target in enumerate(TARGETS):
        for i,fold in enumerate(folds):
            bars=[next(float(x["MAE_meV"]) for x in stats if x["folds"]==fold and x["variant"]=="primary"
                       and x["target"]==target and x["method"]==m) for m in METHODS]
            y=np.arange(3)
            ax[i,j].barh(y,bars,color=colors,height=.65)
            ax[i,j].set_yticks(y,METHODS)
            ax[i,j].invert_yaxis()
            ax[i,j].grid(axis="x",alpha=.2)
            ax[i,j].set_title(f"{estimands[j]}  |  {'LOPO' if i==0 else 'Held-out anion family'}")
            ax[i,j].set_xlabel("Held-out MAE (meV)")
            ax[i,j].set_xlim(0,max(bars)*1.2)
            for yi,bar in zip(y,bars):
                ax[i,j].text(bar+.04,yi,f"{bar:.2f}",va="center",fontsize=9)
    fig.savefig(args.output/"activation_increment_cv.png",dpi=180,bbox_inches="tight",pad_inches=.16)
    plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(11.3,3.9),layout="constrained",sharey=True)
    null=read(args.gpu_results/"blocked_permutations.csv")
    for i,fold in enumerate(folds):
        x=[float(r["SOAP_minus_permuted_combined_MAE_meV"]) for r in null if r["folds"]==fold and r["target"]=="Delta_net"]
        row=next(r for r in uncertainty if r["folds"]==fold and r["target"]=="Delta_net")
        val=float(row["observed_SOAP_minus_combined_MAE_meV"])
        ax[i].hist(x,bins=24,color="#8baec0",edgecolor="white",label="Blocked permutation")
        ax[i].axvline(val,color="#b14a2a",linewidth=2,label=f"Observed, p={float(row['blocked_p_one_sided']):.3f}")
        ax[i].axvline(0,color="#394753",linewidth=1,linestyle="--")
        ax[i].set_title("LOPO" if i==0 else "Held-out anion family")
        ax[i].set_xlabel("SOAP MAE minus combined MAE (meV)")
        ax[i].legend(fontsize=8)
    ax[0].set_ylabel("Permutation count")
    fig.savefig(args.output/"blocked_permutation_net_gain.png",dpi=180,bbox_inches="tight",pad_inches=.16)


if __name__=="__main__":
    main()
