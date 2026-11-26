"""Add separate stratified low-disagreement controls without changing the H1-B gate."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
from statistics import median

from phase_h1_shortlist import read, write


def run(source, supplemental, out):
    policy=json.loads(supplemental.read_bytes())
    statuses=read(source/"candidate_ranking_and_audit.csv")
    main=read(source/"prospective_shortlist.csv")
    if 8<=len(main)<=12:
        maximum=len(main)
    else:
        maximum=policy["targets"]["shortlist_count"]
    selected=[]
    links=defaultdict(set)
    for r in read(source/"parent_structure_matches.csv"):
        if r["cohort"]=="candidate_parent" and r["anonymous_match"]=="True":
            a,b=r["candidate_edge"],r["other_edge"]
            links[a].add(b); links[b].add(a)
    for r in read(source/"li_soap_screen_flags.csv"):
        if r["other_cohort"]=="candidate":
            a,b=r["candidate_edge"],r["other_edge"]
            links[a].add(b); links[b].add(a)
    for item in main:
        selected.append(dict(item,panel_role="high_disagreement"))
    count=Counter(r["anion_family"] for r in selected)
    ids={r["edge_id"] for r in selected}
    pools=defaultdict(list)
    for r in statuses:
        if r["status"]=="insufficient_disagreement":
            median_rank=median(int(v) for k,v in r.items() if k.startswith("rank_") and k!="rank_span")
            pools[r["anion_family"]].append((int(r["rank_span"]),median_rank,r["edge_id"],r))
    for v in pools.values():
        v.sort(key=lambda r:r[:3])
    while len(selected)<maximum:
        possible=[g for g in pools if pools[g] and count[g]<policy["targets"]["maximum_per_broad_family"]]
        if not possible:
            break
        group=min(possible,key=lambda g:(count[g],g))
        spread,central,edge,row=pools[group].pop(0)
        if links[edge]&ids:
            continue
        ids.add(edge)
        count[group]+=1
        selected.append(dict(row,shortlist_position=len(selected)+1,panel_role="low_disagreement_control",
            reason=f"{group}: spread {spread}, median rank {central:g}; control, original high-disagreement gate not passed"))
    write(out/"prospective_review_panel.csv",selected)
    (out/"review_panel_summary.json").write_text(json.dumps(dict(total=len(selected), high_disagreement=len(main),
        control_count=len(selected)-len(main), family_count=dict(count), panel_minimum_met=len(selected)>=8),
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print((out/"review_panel_summary.json").read_text(encoding="utf-8"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","supplemental","out"):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    run(args.source,args.supplemental,args.out)
