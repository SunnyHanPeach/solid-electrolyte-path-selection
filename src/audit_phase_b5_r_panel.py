"""Audit publicly available barrier metadata and endpoint mapping without reading project locks."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from ase.geometry import find_mic
import numpy as np


def cartesian(entry):
    xyz = np.array(entry["coords"])
    return xyz if entry["cartesian"] else xyz @ np.array(entry["lattice_mat"])


def evaluate(entry):
    a,b = entry["structure_ini"],entry["structure_fin"]
    species = a["elements"]
    if species!=b["elements"] or not np.allclose(a["lattice_mat"],b["lattice_mat"],atol=1e-4,rtol=0):
        return "endpoint_cell_or_order_differs",None
    if "Li" not in species:
        return "no_Li",None
    _,dist = find_mic(cartesian(b)-cartesian(a),np.asarray(a["lattice_mat"]),True)
    ii = np.argsort(dist)
    largest,next_largest = dist[ii[-1]],dist[ii[-2]]
    if largest < 0.75 or next_largest > min(0.6, largest/2):
        return "mobility_ambiguous",None
    if species[ii[-1]]!="Li":
        return "non_Li_mover",None
    return "Li_mover_candidate","-".join(sorted(set(species)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("literature_json","historical_index","candidate_manifest","output"):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    source=args.literature_json.read_bytes()
    public=json.loads(source)
    with args.historical_index.open(newline="",encoding="utf-8-sig") as f:
        hist={x["chemsys"] for x in csv.DictReader(f)}
    prospective={x["chemsys"] for x in json.loads(args.candidate_manifest.read_bytes())["rows"] if x["status"]=="included"}
    counts,xc_all,xc_li,xc_disjoint=Counter(),Counter(),Counter(),Counter()
    eligible_systems,eligible_references=set(),set()
    record=[]
    for x in public:
        status,chem=evaluate(x)
        counts[status]+=1
        xc_all[x["XC"]]+=1
        if chem is None:
            continue
        xc_li[x["XC"]]+=1
        if chem in hist or chem in prospective:
            continue
        xc_disjoint[x["XC"]]+=1
        eligible_systems.add(chem)
        eligible_references.add(x["bibtex"])
        record.append(dict(public_id=x["jid"],chemsys=chem,reference_xc=x["XC"],
                           crystal_class=x["crystal_class"],space_group=x["space_group"]))
    result=dict(local_dataset_sha256=sha256(source).hexdigest(), public_records=len(public),
                method="same atomic symbols and cell, minimum-image motion, largest Li >=0.75 A, all other displacements <=max permitted",
                threshold_other="min(0.6 A, half largest displacement)", initial_status=counts,
                exchange_correlation_all=xc_all, exchange_correlation_Li=xc_li,
                exchange_correlation_disjoint_exact_chemistry=xc_disjoint,
                exact_chemistry_disjoint_systems=len(eligible_systems), exact_chemistry_disjoint_source_BibTeX_strings=len(eligible_references),
                disjoint_candidate_metadata=record, release_for_primary_panel=False,
                reserved_project_labels_accessed=0)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print({k:v for k,v in result.items() if k!="disjoint_candidate_metadata"})


if __name__=="__main__":
    main()
