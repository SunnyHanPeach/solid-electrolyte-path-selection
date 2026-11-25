"""Verify the ten-path B2 run from optimizer traces and raw blind predictions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile
import tarfile

import numpy as np

from audit_signed_barriers import file_sha, load_frozen_data, rows

MAIN = (17,29,43)
EXTRA = (53,67)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def table(data):
    return rows(data)


def value_fields(row, names):
    return [float(row[k]) for k in names]


def equal(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-9)


def paired_metrics(materials, intervals, expectations, truth):
    chem = sorted({r["chemsys"] for r in truth.values()})
    n = len(chem)
    per_group_size = Counter(r["chemsys"] for r in truth.values())
    counts = np.random.default_rng(20261126).multinomial(n,np.full(n,1/n),size=10000)
    denominator = counts @ [per_group_size[c] for c in chem]
    for row in intervals:
        kind, slot, metric = row["analysis"],int(row["slot"]),row["metric"]
        wanted, seed_gains = expectations[kind,slot], []
        expected_seeds = ";".join(map(str,wanted))
        assert row["seeds"] == expected_seeds and int(row["materials"]) == len(truth)
        group_totals = defaultdict(list)
        for seed in wanted:
            value = mean(materials[kind,slot,seed,e][metric] for e in truth)
            seed_gains.append(value)
        for e,t in truth.items():
            group_totals[t["chemsys"]].append(mean(materials[kind,slot,s,e][metric] for s in wanted))
        draws = counts @ [math.fsum(group_totals[c]) for c in chem] / denominator
        lo,hi = np.quantile(draws,[.025,.975])
        equal(value_fields(row,("mean_meV","low95_meV","high95_meV")),[mean(seed_gains),lo,hi])
        equal([float(v) for v in row["seed_gains_meV"].split(";")],seed_gains)
        state = ("positive" if lo>.25 and min(seed_gains)>0 else
                 "negative" if hi<-.25 and max(seed_gains)<0 else
                 "near_zero" if lo>=-.25 and hi<=.25 else "unresolved")
        assert row["classification"] == state


def run(args):
    actual = file_sha(args.compact)
    if actual != args.checksum.read_text().strip():
        raise ValueError("Compact download failed SHA256 verification")
    matrix_sha = file_sha(args.matrix)
    with ZipFile(args.matrix) as z:
        plan = json.loads(z.read("plan.json"))
        for name,expected in plan["sha256"].items():
            assert digest(z.read(name))==expected
        jobs = json.loads(z.read("orders.json"))
        base_weights = json.loads(z.read("baselines.json"))
        truth = {r["edge_id"]:r for r in table(z.read("reference_barriers.csv"))}
    assert file_sha(args.base)==plan["base_bundle_sha256"]
    _,_,raw_evaluation = load_frozen_data(args.base,args.project)
    assert len(truth)==107 and len(raw_evaluation)==107
    for (panel, edge),curve in raw_evaluation.items():
        assert panel==truth[edge]["panel"] and len(curve)==int(truth[edge]["images"])
        equal(1000*max(curve),float(truth[edge]["dft_forward_barrier_meV"]))
    with tarfile.open(args.compact,"r:gz") as z:
        members = {m.name:m for m in z if m.isfile()}
        assert all(not n.startswith("/") and ".." not in Path(n).parts for n in members)

        def read(name):
            return z.extractfile(members[name]).read()

        status = json.loads(read("status.json"))
        manifest = json.loads(read("inference_manifest.json"))
        assert status["state"]=="complete" and manifest["matrix_sha256"]==matrix_sha
        slots = status["completed_paths"]
        assert len(slots)==len(set(slots)) and set(slots)<=set(range(1,len(plan["candidates"])+1))
        extra = status["extra_replicated"]
        jobs_scored = {("pure",s) for s in MAIN}
        jobs_scored |= {(f"{a}{i}",s) for a in ("path","replay") for i in slots for s in MAIN}
        expected_fits = 6*len(slots)+4*int(extra)
        if extra:
            jobs_scored |= {(a,s) for a in ("pure","path1","replay1") for s in EXTRA}
        assert status["completed_fits"]==expected_fits
        assert len(manifest["records"])==len(jobs_scored)
        assert {(r["arm"],r["seed"]) for r in manifest["records"]}==jobs_scored
        diag = {(r["arm"],int(r["seed"])):r for r in table(read("training_diagnostics.csv"))}
        assert len(diag)==expected_fits
        compatible = dict(plan.get("compatible_matrix_sha256_slots",{}))
        compatible.setdefault(plan["earlier_matrix_sha256"],6)
        barriers, errors, gaps = {}, {}, []
        for r in manifest["records"]:
            arm,seed = r["arm"],r["seed"]
            if arm=="pure":
                assert r["checkpoint_sha256"]==base_weights[str(seed)]["files"]["model_710.pt"]
            else:
                folder=f"fits/{arm}/seed{seed}"
                marker=json.loads(read(f"{folder}/done.json"))
                trace=read(f"{folder}/training.csv")
                expected=jobs[str(seed)][arm]
                assert marker["schedule_sha256"]==digest(json.dumps(expected).encode())
                assert marker["frozen_steps"]==len(expected) and marker["steps"]==len(expected)
                assert marker["files"]["model.pt"]==r["checkpoint_sha256"]
                assert marker["files"]["training.csv"]==digest(trace)
                slot = int(arm.removeprefix("path").removeprefix("replay"))
                assert marker["matrix_sha256"]==matrix_sha or slot<=compatible.get(marker["matrix_sha256"],0)
                traces=table(trace)
                assert [int(t["step"]) for t in traces]==list(range(1,len(expected)+1))
                per_fit=diag[arm,seed]
                equal(float(per_fit["gradient_clipping_frequency"]),
                      mean(float(t["gradient_norm"])>plan["training"]["clip_norm"] for t in traces))
                if "energy_loss_weighted" in traces[0]:
                    assert per_fit["component_tracking"]=="True"
                    for t in traces:
                        a,b,c=value_fields(t,("loss","energy_loss_weighted","force_loss_weighted"))
                        assert math.isclose(a,b+c,rel_tol=2e-7,abs_tol=1e-8)
                        assert int(t["gradient_clipped"])==int(float(t["gradient_norm"])>plan["training"]["clip_norm"])
                else:
                    assert marker["matrix_sha256"]==plan["earlier_matrix_sha256"]
                    assert per_fit["component_tracking"]=="False"
                    gaps.append(f"{arm}/seed{seed}")
            image=read("predictions/"+r["prediction_file"])
            assert digest(image)==r["prediction_sha256"]
            geometry=defaultdict(list)
            for t in table(image):
                assert t["material_id"]==truth[t["edge_id"]]["material_id"]
                geometry[t["edge_id"]].append((int(t["frame"]),float(t["energy_eV"])))
            assert geometry.keys()==truth.keys()
            for e,curve in geometry.items():
                curve.sort()
                assert [i for i,_ in curve]==list(range(1,int(truth[e]["images"])+1))
                p=1000*(max(v for _,v in curve)-curve[0][1])
                barriers[arm,seed,e]=p
                errors[arm,seed,e]=abs(p-float(truth[e]["dft_forward_barrier_meV"]))
            validation_mae=mean(errors[arm,seed,e] for e in truth)
            equal(validation_mae,r["validation_barrier_mae_meV"])
            if arm!="pure":
                equal(validation_mae,float(diag[arm,seed]["validation_barrier_mae_meV"]))
        distributions={}
        expectations={("regular",slot):MAIN for slot in slots}
        if extra:
            expectations["extra_only",1]=EXTRA
            expectations["five_seed",1]=MAIN+EXTRA
        for r in table(read("material_gains.csv")):
            kind,slot,seed,edge=r["analysis"],int(r["slot"]),int(r["seed"]),r["evaluation_edge"]
            k=kind,slot,seed,edge
            assert k not in distributions
            assert r["candidate_edge"]==plan["candidates"][slot-1]["edge_id"]
            assert r["material_id"]==truth[edge]["material_id"]
            for field,arm in (("pure","pure"),("path",f"path{slot}"),("replay",f"replay{slot}")):
                equal(value_fields(r,(f"{field}_prediction_meV",f"{field}_error_meV")),
                      [barriers[arm,seed,edge],errors[arm,seed,edge]])
            gain={kind_:errors[arm,seed,edge]-errors[f"path{slot}",seed,edge] for kind_,arm in
                  (("absolute","pure"),("net",f"replay{slot}"))}
            equal(value_fields(r,("absolute_gain_meV","net_gain_meV")),[gain["absolute"],gain["net"]])
            distributions[k]=gain
        keys={(kind,slot,seed,e) for (kind,slot),seeds in expectations.items() for seed in seeds for e in truth}
        assert distributions.keys()==keys
        intervals=table(read("intervals.csv"))
        assert {(r["analysis"],int(r["slot"]),r["metric"]) for r in intervals}=={
            (k,s,m) for k,s in expectations for m in ("absolute","net")}
        paired_metrics(distributions,intervals,expectations,truth)
        keep={"inference_manifest.json","intervals.csv","material_gains.csv","checkpoint_hashes.csv",
              "status.json","training_diagnostics.csv"}
        args.output.mkdir(parents=True,exist_ok=True)
        for member in members:
            if member in keep or member.startswith(("fits/","instrumentation_smoke/")):
                dst=args.output/member
                dst.parent.mkdir(parents=True,exist_ok=True)
                with dst.open("xb") as f:
                    f.write(read(member))
    start=datetime.fromisoformat(status["prior_start_utc"].replace("Z","+00:00"))
    finish=datetime.fromisoformat(status["ended_utc"].replace("Z","+00:00"))
    scoring=datetime.fromisoformat(status["scoring_started_utc"].replace("Z","+00:00"))
    result=dict(status="pass", compact_sha256=actual, matrix_sha256=matrix_sha, complete_fits=expected_fits,
                complete_paths=len(slots), material_rows=len(distributions), evaluation_materials=len(truth),
                paired_intervals=len(intervals), legacy_component_unavailable=gaps,
                elapsed_seconds=(finish-start).total_seconds(), scoring_seconds=(finish-scoring).total_seconds())
    with (args.output/"verification.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("compact","checksum","matrix","base","project","output"):
        p.add_argument(key, type=Path)
    run(p.parse_args())
