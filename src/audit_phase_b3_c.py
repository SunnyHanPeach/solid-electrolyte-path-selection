"""Audit short-contact confounding with fixed path-level rank projections."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from phase_a0_geometry import export_csv
from phase_b1_inference import file_sha, read_json, rows, strict_check, write_json


def pearson(x,y):
    if len(x)<3 or min(np.ptp(x),np.ptp(y))<=1e-12:
        return None
    return float(np.corrcoef(x,y)[0,1])


def values(records,feature):
    return np.array([r[feature] for r in records],dtype=float)


def project(x,controls):
    z = np.column_stack([np.ones(len(x)),*controls])
    if len(x)<len(controls)+3 or np.linalg.matrix_rank(z)!=z.shape[1]:
        return None
    return x-z@np.linalg.lstsq(z,x,rcond=None)[0]


def association(records,x,y,controls):
    xr,yr = [rankdata(values(records,k)) for k in (x,y)]
    z = [rankdata(values(records,k)) for k in controls]
    raw = pearson(xr,yr)
    rx,ry = project(xr,z),project(yr,z)
    partial = None if rx is None or ry is None else pearson(rx,ry)
    semi = None if rx is None else pearson(rx,yr)
    return raw,partial,semi


def section(records,label,x,y,controls):
    raw,partial,semi = association(records,x,y,controls)
    return dict(sample=label,feature=x,outcome=y,controls=";".join(controls),
                paths=len(records),marginal_rho=raw,partial_rho=partial,semipartial_rank_r=semi)


def read_data(repo,protocol):
    supplement = repo/"results/phase_b3_a_ensemble"
    features = supplement/"path_feature_matrix.csv"
    labels = repo/"results/phase_b3_a/path_utility_labels.csv"
    manifest = read_json(supplement/"analysis_manifest.json")
    paths_sha = manifest["analysis_sha256"]["path_feature_matrix.csv"]
    gains_sha = manifest["source_B3A_labels_sha256"]
    strict_check(file_sha(features)==paths_sha and file_sha(labels)==gains_sha,"Frozen B3 inputs changed")
    keyed = {r["edge_id"]:r for r in rows(labels)}
    out = []
    for r in rows(features):
        y = keyed[r["edge_id"]]
        strict_check(r["chemsys"]==y["chemsys"] and y["seeds"]=="17;29;43" and y["materials"]=="107",
                     "Frozen label keys or protocol differ")
        fields = {k:float(r[v]) for k,v in protocol["features"].items()}
        fields.update({k:float(y[k]) for k in protocol["outcomes"]})
        strict_check(np.isfinite(list(fields.values())).all(),"Missing variable")
        out.append(dict(edge_id=r["edge_id"],chemsys=r["chemsys"],cohort=r["cohort"],**fields))
    strict_check(len(out)==len(keyed)==14 and {r["edge_id"] for r in out}==set(keyed),"Incorrect cohort")
    return sorted(out,key=lambda r:(r["contact"],r["edge_id"])),paths_sha,gains_sha


def paired_contacts(data,outcomes):
    pairs = []
    for j in range(0,len(data),2):
        a,b = sorted(data[j:j+2],key=lambda r:(r["tangent"],r["edge_id"]))
        item = dict(block=j//2+1,lower_tangent_edge=a["edge_id"],higher_tangent_edge=b["edge_id"],
                    lower_tangent_images=int(a["images"]),higher_tangent_images=int(b["images"]),
                    low_tangent_contact_A=a["contact"],high_tangent_contact_A=b["contact"],
                    absolute_contact_gap_A=abs(b["contact"]-a["contact"]),
                    tangent_difference_eV_per_A=b["tangent"]-a["tangent"])
        for y in outcomes:
            change = b[y]-a[y]
            item[f"{y}_difference_meV"] = change
            item[f"{y}_direction"] = int(np.sign(change)) if item["tangent_difference_eV_per_A"]>0 else 0
        pairs.append(item)
    return pairs


def render(out,data,residuals):
    fig,axs = plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    colors = ["#0072B2" if r["images"]==5 else "#D55E00" for r in data]
    axs[0].scatter(values(data,"contact"),values(data,"tangent"),c=colors)
    axs[0].set(xlabel="Minimum Li contact (Å)",ylabel="Mean tangential force SD (eV/Å)")
    rx = [r["tangent_rank_residual"] for r in residuals]
    for ax,y in zip(axs[1:],("Delta_abs","Delta_net")):
        ax.scatter(rx,[r[f"{y}_rank_residual"] for r in residuals],c=colors)
        ax.axvline(0,color="#888",linewidth=.7)
        ax.axhline(0,color="#888",linewidth=.7)
        ax.set(xlabel="Tangential force rank residual",ylabel=f"{y} rank residual")
    for ax in axs:
        ax.grid(alpha=.16)
    fig.supxlabel("Blue: 5 images. Orange: 7 images. Right panels remove ranked minimum contact.")
    fig.savefig(out/"short_contact_residuals.png",dpi=165)
    plt.close(fig)


def main(repo,out):
    protocol = read_json(out/"analysis_protocol.json")
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    subprocess.run(["git","merge-base","--is-ancestor",protocol["source_commit"],revision],cwd=repo,check=True)
    data,xsha,ysha = read_data(repo,protocol)
    cases = [(x,cs) for x,cs in protocol["primary"]+protocol["sensitivity"]]
    outcomes = protocol["outcomes"]
    complete = [section(data,"all",x,y,c) for x,c in cases for y in outcomes]
    # Cross-check one-control partial Spearman by the three-pair closed form.
    for y in outcomes:
        rxy = association(data,"tangent",y,[])[0]
        rxz = association(data,"tangent","contact",[])[0]
        ryz = association(data,y,"contact",[])[0]
        reference = (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))
        np.testing.assert_allclose(association(data,"tangent",y,["contact"])[1],reference,rtol=0,atol=1e-12)
    leave = [dict(omitted_edge=r["edge_id"],**section([q for q in data if q["edge_id"]!=r["edge_id"]],
                 "leave_one",x,y,c)) for r in data for x,c in cases for y in outcomes]
    strata = {"shorter_half":data[:7],"longer_half":data[7:],
              "five_images":[r for r in data if r["images"]==5],
              "seven_images":[r for r in data if r["images"]==7]}
    split = []
    for label,group in strata.items():
        for y in outcomes:
            for c in ([],["contact"]):
                split.append(dict(section(group,label,"tangent",y,c),
                                  contact_min_A=min(r["contact"] for r in group),
                                  contact_max_A=max(r["contact"] for r in group)))
    pairs = paired_contacts(data,outcomes)
    z = rankdata(values(data,"contact"))
    ranks = {k:project(rankdata(values(data,k)),[z]) for k in ("tangent",*outcomes)}
    raw_force = project(values(data,"tangent"),[values(data,"contact")])
    residuals = [dict(edge_id=r["edge_id"],images=int(r["images"]),contact_A=r["contact"],
                      contact_rank=float(z[i]),tangent_raw_residual_eV_per_A=float(raw_force[i]),
                      **{f"{k}_rank_residual":float(v[i]) for k,v in ranks.items()}) for i,r in enumerate(data)]
    relations = [section(data,"all",x,y,[]) for x,y in (("tangent","contact"),("SOAP","contact"),
                    ("SOAP","tangent"),("SOAP","RDF"),("Activation","tangent"))]
    checks = [dict(outcome=y,paths=len(data),raw_force_residual_spearman=pearson(rankdata(raw_force),
                    rankdata(values(data,y))),rank_semipartial=association(data,"tangent",y,["contact"])[2],
                    partial_spearman=association(data,"tangent",y,["contact"])[1]) for y in outcomes]
    for name,content in (("paired_inputs.csv",data),("rank_associations.csv",complete),
                         ("leave_one_path_out.csv",leave),("stratified_associations.csv",split),
                         ("near_contact_pairs.csv",pairs),("feature_relations.csv",relations),
                         ("residuals.csv",residuals),("residual_associations.csv",checks)):
        export_csv(out/name,content)
    render(out,data,residuals)
    write_json(out/"analysis_manifest.json",dict(source_commit=revision,
        feature_input_sha256=xsha,label_input_sha256=ysha,script_sha256=file_sha(Path(__file__)),
        protocol_sha256=file_sha(out/"analysis_protocol.json"),
        versions={p:version(p) for p in ("numpy","scipy","matplotlib")},paths=len(data),
        one_control_formula_verified=True,
        output_sha256={p.name:file_sha(p) for p in sorted(out.iterdir()) if p.suffix in (".csv",".png")}))
    print(f"Verified B3-C on {len(data)} paths, {len(complete)} frozen specifications, {len(leave)} leave-one rows")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path)
    p.add_argument("out",type=Path)
    main(**vars(p.parse_args()))
