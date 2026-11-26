"""Cloud structural overlap audit and chemistry-stratified H1 shortlist."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from hashlib import sha256
from itertools import combinations
import json
from math import gcd
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

from ase.io import read as ase_read
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=()):
    fields=list(dict.fromkeys([*fields,*[k for row in rows for k in row]]))
    with path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fields)
        w.writeheader()
        w.writerows(rows)


def read_parents(archive_path, ids):
    raw_sha=sha256(archive_path.read_bytes()).hexdigest()
    if raw_sha!="3d89f921c47101bd3c9050e92829813e57f05b8178ef5bc3ce16dfb58b845c22":
        raise ValueError("BVEL source archive differs")
    output={}
    pattern=re.compile(rb"(?:^| )material_id=([^\s]+)")
    with ZipFile(archive_path) as archive:
        for split in ("train","val","test"):
            with archive.open(f"BVEL13k/BVEL13k_{split}.xyz") as stream:
                while n:=stream.readline():
                    if not n.strip():
                        continue
                    header=stream.readline()
                    key=pattern.search(header)[1].decode()
                    lines=[stream.readline() for _ in range(int(n))]
                    if key not in ids:
                        continue
                    if key in output:
                        raise ValueError(f"Repeated parent for {key}")
                    geometry=header.split(b'Lattice="',1)[1].split(b'"',1)[0]
                    z=[line.split() for line in lines]
                    species=[r[0].decode() for r in z]
                    xyz=[[float(x) for x in r[1:4]] for r in z]
                    lattice=np.fromstring(geometry.decode(),sep=" ").reshape(3,3)
                    output[key]=(Structure(lattice,species,xyz,coords_are_cartesian=True),
                                 sha256(b"".join([n,header,*lines])).hexdigest())
    if set(output)!=ids:
        raise ValueError(f"Missing historic or candidate parents: {sorted(ids-set(output))}")
    return output,raw_sha


def anonymous_formula(structure):
    counts=list(Counter(s.symbol for s in structure.species).values())
    divisor=gcd(*counts)
    return tuple(sorted(x//divisor for x in counts))


def crystal_audit(candidates, historical, parents, settings, workers):
    hist_mats={r["material_id"] for r in historical}
    newer={r["material_id"]:r["edge_id"] for r in candidates}
    grouped=defaultdict(list)
    for material,(s,_) in parents.items():
        grouped[anonymous_formula(s)].append(material)
    candidate_pairs=[]
    for material,edge in newer.items():
        for other in sorted(grouped[anonymous_formula(parents[material][0])]):
            if other in newer and other<=material:
                continue
            if other!=material:
                candidate_pairs.append((material,other,edge))
    def match(pair):
        material,other,edge=pair
        matcher=StructureMatcher(ltol=settings["ltol"],stol=settings["stol"],angle_tol=settings["angle_tol_deg"],
                                 primitive_cell=settings["primitive_cell"],scale=settings["scale"])
        a,b=parents[material][0],parents[other][0]
        yes=matcher.fit_anonymous(a,b)
        return dict(candidate_edge=edge,other_material=other,other_edge=newer.get(other,""),
                    cohort="historical_parent" if other in hist_mats else "candidate_parent",
                    anonymous_match=bool(yes),same_elementset=(set(a.symbol_set)==set(b.symbol_set)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(match,candidate_pairs))


def signature(symbols):
    return "-".join(sorted(set(symbols)))


def radial_audit(data,candidates,historical,plan):
    c_edges={r["edge_id"]:r for r in candidates}
    hist_info={r["edge_id"]:r for r in historical}
    cs={edge:signature(sym for sym,q in row["oxidation_states"] if q<0) for edge,row in c_edges.items()}
    history_curves=defaultdict(list)
    with gzip.open(data/"historical_anion_neighbors.csv.gz","rt",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            history_curves[row["edge_id"]].append((int(row["frame"]),
                    [float(row[f"anion_neighbor_{i}_A"]) for i in range(1,5)]))
    if set(history_curves)!=set(hist_info):
        raise ValueError("Historical path local coverage differs")
    by_group=defaultdict(list)
    for edge, raw in history_curves.items():
        curve=np.asarray([x for _,x in sorted(raw)])
        if len(curve)!=int(hist_info[edge]["init_images"]):
            raise ValueError("Historical local image count differs")
        by_group[hist_info[edge]["anion_family"]].append((edge, sample(curve),float(hist_info[edge]["hop_displacement_A"])))
    new_curve=defaultdict(list)
    mover_positions=defaultdict(list)
    for frame in ase_read(data/"candidate_images.xyz",":",format="extxyz"):
        e=frame.info["edge_id"]
        number=int(frame.info["moving"])
        take=[i for i,s in enumerate(frame.get_chemical_symbols()) if s in cs[e].split("-")]
        if len(take)<4:
            continue
        new_curve[e].append(np.sort(frame.get_distances(number,take,mic=True))[:4])
        mover_positions[e].append((frame.positions[number].copy(),frame.cell))
    radial, scope, hops = [],[],{}
    from ase.geometry import find_mic
    for edge, values in new_curve.items():
        positions=np.array([x for x,_ in mover_positions[edge]])
        step,_=find_mic(np.diff(positions,axis=0),mover_positions[edge][0][1],True)
        hops[edge]=float(np.linalg.norm(step.sum(axis=0)))
        curve=sample(np.array(values))
        candidates_group=by_group[cs[edge]]
        closest=None
        n=0
        for other,ref,hop in candidates_group:
            d_hop=abs(hops[edge]-hop)
            if d_hop>plan["max_hop_difference_A"]:
                continue
            n+=1
            forward=float(np.sqrt(np.mean((curve-ref)**2)))
            reverse=float(np.sqrt(np.mean((curve-ref[::-1])**2)))
            delta=min(forward,reverse)
            pair=dict(edge_id=edge,historical_edge=other,anion_signature=cs[edge],delta_hop_A=d_hop,
                      radial_rms_A=delta,reversed_orientation=reverse<forward)
            if closest is None or delta<closest["radial_rms_A"]:
                closest=pair
            if delta<=plan["max_path_mean_rms_A"]:
                radial.append(pair)
        scope.append(dict(edge_id=edge,available_same_anions=len(candidates_group),compared_same_hop=n,
                          nearest_edge=closest["historical_edge"] if closest else "",
                          nearest_rms_A=closest["radial_rms_A"] if closest else None))
    return radial,scope,cs,hops


def sample(curve):
    # The historical metadata supplies frame order only. This comparison interpolates the image index.
    old=np.linspace(0,1,len(curve))
    return np.array([np.interp(np.linspace(0,1,5),old,v) for v in curve.T]).T


def displacement_from_profiles(rows):
    # ASE mobile coordinates are authoritative if available. This routine is for candidates without four anions.
    from ase.geometry import find_mic
    positions=defaultdict(list)
    cells={}
    for a in rows:
        edge=a.info["edge_id"]
        positions[edge].append(a.positions[int(a.info["moving"])].copy())
        cells[edge]=a.cell
    return {edge:float(np.linalg.norm(find_mic(np.diff(np.array(obs),axis=0),cells[edge],True)[0].sum(axis=0)))
            for edge,obs in positions.items()}


def shortlist(data,gpu,out,protocol,parents,structures,radial,radial_scope,signatures,hops):
    frozen=read(data/"candidate_selection_audit.csv")
    cands={r["edge_id"]:r for r in frozen if r["status"]=="included"}
    old={r["edge_id"]:r for r in read(data/"historical_geometry_index.csv")}
    ranks=defaultdict(dict)
    for row in read(data/"independent_candidate_ranking.csv"):
        if row["aggregation"]=="mean" and row["method"] in protocol["strategies"]:
            ranks[row["edge_id"]][row["method"]]=int(row["rank"])
    if set(cands)!=set(ranks) or any(len(r)!=len(protocol["strategies"]) for r in ranks.values()):
        raise ValueError("Frozen rank rows unexpectedly changed")
    old_mat={r["material_id"] for r in old.values()}
    old_systems={r["chemsys"] for r in old.values()}
    hold,links,li_matches=defaultdict(list),defaultdict(set),[]
    for row in structures:
        if not row["anonymous_match"]:
            continue
        e=row["candidate_edge"]
        if row["cohort"]=="historical_parent":
            hold[e].append(f"anonymous-parent match: {row['other_material']}")
        else:
            o=row["other_edge"]
            links[e].add(o)
            links[o].add(e)
    for row in read(gpu/"li_soap_curve_distances.csv"):
        a,b=row["edge_a"],row["edge_b"]
        if a not in cands and b not in cands:
            continue
        if b in cands and a not in cands:
            a,b=b,a
        same=signatures[a]==(signatures[b] if b in cands else old[b]["anion_family"])
        d_hop=abs(hops[a]-(hops[b] if b in cands else float(old[b]["hop_displacement_A"])))
        spec=protocol["li_soap_confirm"]
        if not same or d_hop>spec["max_hop_difference_A"] or float(row["cosine_distance"])>spec["max_mean_cosine_distance"]:
            continue
        li_matches.append(dict(candidate_edge=a,other_edge=b,other_cohort="candidate" if b in cands else "historical_dev",
                               delta_hop_A=d_hop,cosine_distance=float(row["cosine_distance"]),
                               reversed_orientation=row["reversed_orientation"]))
        if b in cands:
            links[a].add(b); links[b].add(a)
        else:
            hold[a].append(f"same-anion Li SOAP curve match: {b}")
    write(out/"li_soap_screen_flags.csv",li_matches,
          ("candidate_edge","other_edge","other_cohort","delta_hop_A","cosine_distance","reversed_orientation"))
    strata=defaultdict(list)
    audit={}
    for edge,row in cands.items():
        if row["material_id"] in old_mat or row["chemsys"] in old_systems:
            hold[edge].append("historical material ID or exact chemistry")
        if parents[row["material_id"]][1]!=row["parent_sha256"]:
            raise ValueError("B5-R frozen parent hash differs")
        ranking=ranks[edge]
        best,worst=min(ranking.values()),max(ranking.values())
        span=worst-best
        top=[name for name,r in ranking.items() if r<=5]
        group=row["材料家族"]
        entry=dict(edge_id=edge,material_id=row["material_id"],chemsys=row["chemsys"],anion_family=group,
                   crystal_family_proxy=row["family"],**{f"rank_{k}":ranking[k] for k in protocol["strategies"]},
                   best_rank=best,worst_rank=worst,rank_span=span,top5_strategies=";".join(top),
                   radial_historical_flags=sum(x["edge_id"]==edge for x in radial),atoms=row["atoms"],images=row["images"],
                   structural_exclusion=" | ".join(sorted(hold[edge])),linked_candidate_count=len(links[edge]))
        audit[edge]=entry
        if hold[edge]:
            entry["status"]="structural_exclusion"
        elif span<protocol["ranking_disagreement_min_span"] or best>protocol["best_rank_max"]:
            entry["status"]="insufficient_disagreement"
        else:
            entry["status"]="eligible"
            strata[group].append((not (0<len(top)<len(ranking)),-span,best,edge))
    for values in strata.values():
        values.sort()
    selected=[]
    for depth in range(max(map(len,strata.values()),default=0)):
        for group, entries in sorted(strata.items()):
            if depth>=len(entries) or len(selected)>=protocol["shortlist_target"]:
                continue
            edge=entries[depth][-1]
            if sum(audit[e]["anion_family"]==group for e in selected)>=protocol["family_cap"]:
                audit[edge]["status"]="family_cap"
            elif links[edge].intersection(selected):
                audit[edge]["status"]="duplicate_of_selected"
            else:
                selected.append(edge)
                ranking=ranks[edge]
                favored=min(ranking,key=ranking.get)
                disfavored=max(ranking,key=ranking.get)
                audit[edge].update(status="selected",shortlist_position=len(selected),
                                   reason=f"{group}: {favored} rank {ranking[favored]} vs {disfavored} rank {ranking[disfavored]}")
    write(out/"candidate_ranking_and_audit.csv",list(audit.values()))
    write(out/"prospective_shortlist.csv",[audit[e] for e in selected],("shortlist_position","edge_id"))
    return selected,audit,li_matches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data","gpu","protocol","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=8)
    args=p.parse_args()
    start=perf_counter()
    data,out=args.data,args.out
    out.mkdir()
    protocol=json.loads(args.protocol.read_bytes())["candidate"]
    raw=json.loads((data/"candidate_manifest.json").read_bytes())
    if sha256((data/"candidate_images.xyz").read_bytes()).hexdigest()!=json.loads((data/"b5_manifest.json").read_bytes())["geometry_sha256"]["candidate_images.xyz"]:
        raise ValueError("Candidate trajectories differ from B5-R")
    candidates=[r for r in raw["rows"] if r["status"]=="included"]
    historical=read(data/"historical_geometry_index.csv")
    if len(historical)!=1681 or len(candidates)!=24:
        raise ValueError("Unexpected candidate/source row counts")
    parents,archive_sha=read_parents(data/"BVEL13k.zip",{r["material_id"] for r in historical+candidates})
    structure=crystal_audit(candidates,historical,parents,protocol["parent_anonymous_match"],args.workers)
    write(out/"parent_structure_matches.csv",structure)
    radial,scope,signatures,hops=radial_audit(data,candidates,historical,protocol["local_four_anion_screen"])
    if len(hops)!=len(candidates):
        hops=displacement_from_profiles(ase_read(data/"candidate_images.xyz",":",format="extxyz"))
    write(out/"radial_historical_flags.csv",radial,
          ("edge_id","historical_edge","anion_signature","delta_hop_A","radial_rms_A","reversed_orientation"))
    write(out/"radial_historical_coverage.csv",scope)
    selected,audit,li=shortlist(data,args.gpu,out,protocol,parents,structure,radial,scope,signatures,hops)
    result=dict(total_parents=len(parents),historical_materials=len({r["material_id"] for r in historical}),
                parents_archive_sha256=archive_sha, anonymous_compared=len(structure),
                anonymous_confirmed=sum(r["anonymous_match"] for r in structure),radial_flags=len(radial),
                radial_candidate_coverage=len(scope),li_soap_matches=len(li),
                status=Counter(r["status"] for r in audit.values()),shortlist_count=len(selected),
                shortlist_families=Counter(audit[e]["anion_family"] for e in selected),
                minimum_met=len(selected)>=protocol["minimum"],workers=args.workers,elapsed_seconds=perf_counter()-start)
    (out/"structure_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
