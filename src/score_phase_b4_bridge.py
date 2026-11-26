"""Check fixed geometry VASP/QE bridge outputs without running any calculation."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ase.io import read
from ase.units import Bohr, Ry


FORCE_RE = re.compile(r"^\s*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s*$", re.M)
TOTAL_RE = re.compile(r"^!\s+total energy\s*=\s*(\S+)\s+Ry", re.M)
SMEAR_RE = re.compile(r"^\s*smearing contrib\.\s*\(-TS\)\s*=\s*(\S+)\s+Ry", re.M)
ENERGY_GATE_ME_V=10.0
FORCE_GATE_EV_A=0.02


def fnum(txt):
    return float(txt.replace("D", "E").replace("d", "e"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def per_image(ref, folder, moving):
    state=json.loads((folder/"state.json").read_text())
    row={"image":state["image"], "solver_status":state["status"],"ranks":state["ranks"]}
    for key in ("wall_seconds", "rank_seconds", "gpu_seconds", "device", "qe_version",
                "sampled_peak_aggregate_rss_KiB", "exit_code"):
        if key in state:
            row[key]=state[key]
    if state["status"] != "completed":
        return row

    log_bytes=(folder/"pw.out").read_bytes()
    log=log_bytes.decode(errors="strict")
    assert "JOB DONE." in log and "convergence has been achieved" in log
    rows=FORCE_RE.findall(log)
    n=len(ref)
    assert len(rows)>=n
    assert [int(i) for i,*_ in rows[-n:]]==list(range(1,n+1))
    forces=np.array([[fnum(x),fnum(y),fnum(z)] for _,x,y,z in rows[-n:]])*(Ry/Bohr)
    ref_force=ref.get_forces()
    d=forces-ref_force
    e=fnum(TOTAL_RE.findall(log)[-1])*Ry
    entropy=SMEAR_RE.findall(log)
    qe_smearing=fnum(entropy[-1])*Ry if entropy else None
    ref_moving=ref_force[moving]
    qe_moving=forces[moving]
    norms=np.linalg.norm(ref_moving)*np.linalg.norm(qe_moving)
    cosine=float(np.dot(ref_moving,qe_moving)/norms) if min(np.linalg.norm(ref_moving),np.linalg.norm(qe_moving))>=.01 else None
    row.update(qe_total_energy_eV=e, qe_smearing_term_eV=qe_smearing,
        vasp_free_energy_eV=float(ref.get_potential_energy(force_consistent=True)),
        vasp_potential_energy_eV=float(ref.get_potential_energy()),
        force_component_rmse_eV_A=float(np.sqrt(np.mean(d**2))),
        per_element_force_rmse_eV_A={
            el:float(np.sqrt(np.mean(d[np.array(ref.get_chemical_symbols())==el]**2)))
            for el in sorted(set(ref.get_chemical_symbols()))
        },
        migrating_li_qe_force_eV_A=qe_moving.tolist(), migrating_li_vasp_force_eV_A=ref_moving.tolist(),
        migrating_li_force_difference_norm_eV_A=float(np.linalg.norm(d[moving])),
        migrating_li_force_cosine=cosine,
        qe_log_sha256=sha(log_bytes),
        qe_input_sha256=state["executed_sha256"])
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("runs",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    cfg=json.loads(args.manifest.read_text())
    with ZipFile(args.archive) as z:
        source=z.read(cfg["source_member"])
    assert sha(source)==cfg["source_sha256"]
    refs=read(io.StringIO(source.decode()),":",format="extxyz")
    moving=cfg["moving_Li_0based"]
    result=[]
    for i,ref in enumerate(refs):
        d=args.runs/f"f{i:02d}"
        row=per_image(ref,d,moving) if (d/"state.json").is_file() else {"image":i,"solver_status":"not_started"}
        result.append(row)
    complete=[x for x in result if "qe_total_energy_eV" in x]
    zero=next((r for r in complete if r["image"]==0),None)
    if zero is not None:
        for row in complete:
            dqe=row["qe_total_energy_eV"]-zero["qe_total_energy_eV"]
            dvasp=row["vasp_free_energy_eV"]-zero["vasp_free_energy_eV"]
            row.update(qe_relative_energy_meV=1000*dqe, vasp_relative_free_energy_meV=1000*dvasp,
                        free_energy_profile_difference_meV=1000*(dqe-dvasp))
            if row["qe_smearing_term_eV"] is not None and zero["qe_smearing_term_eV"] is not None:
                eq0=row["qe_total_energy_eV"]-.5*row["qe_smearing_term_eV"]
                eb0=zero["qe_total_energy_eV"]-.5*zero["qe_smearing_term_eV"]
                ev0=row["vasp_potential_energy_eV"]-zero["vasp_potential_energy_eV"]
                row["gaussian_zero_smearing_profile_difference_meV"]=1000*(eq0-eb0-ev0)
    ids={x["image"] for x in complete}
    stage1_done={0,3,4}<=ids
    all_done=len(complete)==len(refs)
    worst_force=max((x["force_component_rmse_eV_A"] for x in complete),default=None)
    energy=max((abs(x["free_energy_profile_difference_meV"]) for x in complete
                if "free_energy_profile_difference_meV" in x),default=None)
    within=(energy is not None and energy<=ENERGY_GATE_ME_V and worst_force<=FORCE_GATE_EV_A)
    peak={}
    peak_ok=False
    if complete:
        q=max(complete,key=lambda x:x["qe_total_energy_eV"])
        v=max(complete,key=lambda x:x["vasp_free_energy_eV"])
        match=q["image"]==v["image"]
        gap_q=1000*(q["qe_total_energy_eV"]-v["qe_total_energy_eV"])
        gap_v=1000*(v["vasp_free_energy_eV"]-q["vasp_free_energy_eV"])
        near=not match and max(gap_q,gap_v)<=ENERGY_GATE_ME_V
        peak_ok=match or near
        peak={"qe_observed_peak_image":q["image"],"vasp_observed_peak_image":v["image"],
              "observed_peak_match":match,"observed_peak_near_degenerate":near,
              "peak_choice_gaps_meV":{"QE":gap_q,"VASP":gap_v}}
    output={"energy_convention":"primary QE finite-smearing total vs VASP free_energy; secondary Gaussian zero-smearing estimate",
            "energy_gate_meV":ENERGY_GATE_ME_V,"force_component_rmse_gate_eV_A":FORCE_GATE_EV_A,
            "worst_profile_difference_meV":energy,"worst_force_rmse_eV_A":worst_force,
            "stage1_done":stage1_done,"all_five_done":all_done,
            "completed_images_within_gates":within if complete else None,
            "material_bridge_quantitative_pass":bool(all_done and within and peak_ok),
            "total_observed_assigned_rank_hours":sum(r.get("rank_seconds",0.) for r in result)/3600.,
            "total_observed_gpu_hours":sum(r.get("gpu_seconds",0.) for r in result)/3600.,
            "image_peaks_on_observed_subset":peak,"images":result}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k!="images"}))


if __name__=="__main__":
    main()
