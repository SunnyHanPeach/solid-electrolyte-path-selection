"""Read checkpoint bytes, metadata and archived predictions without loading a model."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import pickletools
import tarfile
from zipfile import ZipFile, is_zipfile

FOUNDATION = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"
SEEDS = (17,29,43)
NET = "results/net_path_panel"
PHASE = "results/phase1"


def digest(data):
    return sha256(data).hexdigest()


def records(blob):
    return list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))


def frame_prefix(blob, n):
    f = io.BytesIO(blob)
    for _ in range(n):
        atoms = int(f.readline())
        if any(not f.readline() for _ in range(atoms+1)):
            raise ValueError("Truncated D0 frame")
    return blob[:f.tell()]


def checkpoint_type(blob):
    buffer = io.BytesIO(blob)
    if not is_zipfile(buffer):
        return "unclassified non-ZIP serialization"
    with ZipFile(buffer) as z:
        pickle_name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        globals_ = [arg for op,arg,_ in pickletools.genops(z.read(pickle_name)) if op.name=="GLOBAL"]
    global_models = [g.replace(" ",".") for g in globals_ if g.startswith("mace.modules.models ")]
    return global_models[0] if global_models else "state_dict; class inherited from foundation"


def write_table(out, name, rows):
    with (out/name).open("x",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def audit(args):
    repo, store, history, project = args.repo,args.archives,args.history,args.project
    published = history/"outputs"
    records_by_id, hashes, provenance = [], [], []
    pack = store/"phase1-p0/bundle.zip"
    with ZipFile(pack) as z:
        phase1 = json.loads(z.read("plan.json"))
        canonical = frame_prefix(z.read("training.xyz"),278)
        official_blob = z.read("foundation.model")
        if digest(official_blob)!=FOUNDATION:
            raise ValueError("Wrong foundation in the Phase 1 package")
    with ZipFile(store/"MACE_batch_pilot_GPU_bundle.zip") as z:
        batch = json.loads(z.read("execution.json"))
        if frame_prefix(z.read("training.xyz"),278)!=canonical:
            raise ValueError("Batch control does not use the canonical D0 frames")
    deploy = published/"MACE_106_panel_reconstruction_deploy.zip"
    with ZipFile(deploy) as z:
        design = z.read("source/design.json")
        info = json.loads(design)
        if info["d0"] != list(range(278)) or frame_prefix(z.read("source/geometries.extxyz"),278)!=canonical:
            raise ValueError("Historical archive does not match the canonical D0")
        bi_jobs = {(r["arm"],int(r["seed"])):r for r in records(z.read("jobs.csv"))
                   if r["pair_key"]=="v2.0/pair_1"}
        bi_extra = digest(z.read("v2/extra.xyz"))
    with ZipFile(published/"MACE_E0_clip_factorial_v1.zip") as z:
        config = json.loads(z.read("protocol.json"))["protocol"]
        if config["design_sha256"]!=digest(design):
            raise ValueError("Factorial design differs")
    data_identity = digest(canonical)

    def add(identity, category, group, path, marker, tape, expected, blob, settings, data,
            panel, scoring, train_logs, notes, eligibility="excluded", order="", copies=""):
        real = digest(blob) if blob is not None else ""
        if real and real!=expected:
            raise ValueError(f"Hash mismatch for {identity}")
        availability = "marker only; model bytes unavailable"
        if real:
            availability = "local bytes present; provenance unlinked" if category=="D" else "local hashes match"
        records_by_id.append(dict(checkpoint_id=identity, class_id=category, protocol_group=group,
                                  sha256_recorded=expected, file=path, architecture=tape,
                                  availability=availability,
                                  ensemble_eligibility=eligibility, identical_local_copy=copies))
        hashes.append(dict(checkpoint_id=identity, path=path, sha256_expected=expected, sha256_computed=real,
                           verification="byte-verified" if real else "not verified against checkpoint bytes"))
        if copies:
            hashes.append(dict(checkpoint_id=identity,path=copies,sha256_expected=expected,
                               sha256_computed=real,verification="byte-identical copy"))
        environment = marker.get("environment",marker.get("runtime",{}))
        is_fit = bool(settings)
        provenance.append(dict(
            checkpoint_id=identity, initialization_sha256=FOUNDATION if is_fit else "",
            mace_version=environment.get("mace","not recorded for upstream serialization"),
            hardware=environment.get("hardware",environment.get("gpu","")),
            driver=environment.get("driver",""), torch_version=environment.get("torch",""),
            training_data=data, canonical_D0_xyz_sha256=data_identity if is_fit else "",
            optimizer_steps=marker.get("steps",marker.get("optimizer_steps","")), seed=marker.get("seed",""),
            energy_weight=settings.get("energy_weight",""),forces_weight=settings.get("forces_weight",""),
            learning_rate=settings.get("lr",""),clip_norm=settings.get("clip_norm",""),
            batch_size=settings.get("batch_size",""),optimizer="Adam" if is_fit else "",
            optimizer_state="not archived" if is_fit else "not applicable",
            order_sha256=order, training_recipe_source=notes, training_log_status=train_logs,
            evaluation_panel=panel, scoring_records=scoring
        ))

    architecture = checkpoint_type(official_blob)
    add("foundation-small","C","original-MACE-MP-0",
        "archives/phase1-p0/bundle.zip!foundation.model",{},architecture,FOUNDATION,official_blob,{},
        "upstream foundation; project data do not define upstream training","A1 uses fine-tuned D0 not this file",
        "", "upstream not audited","project frozen source initialization")
    alternatives = (
        ("project",project,"work/mace-pair/mace-mp-0-small.model"),
        ("history",history,"work/mace-pair/mace-mp-0-small.model"),
        ("history",history,"work/label-diagnostics/panel_cpu_blind_reconstruction/mace-mp-0-small.model"),
    )
    for ordinal,(alias,base,path) in enumerate(alternatives,1):
        candidate = base/path
        if not candidate.is_file():
            continue
        raw = candidate.read_bytes()
        if digest(raw)==FOUNDATION:
            records_by_id[0]["identical_local_copy"] += f" {alias}/{path}"
            hashes.append(dict(checkpoint_id="foundation-small",path=f"{alias}/{path}",
                               sha256_expected=FOUNDATION,sha256_computed=FOUNDATION,verification="byte-identical copy"))
            continue
        declared = digest(raw)
        add(f"foundation-unlinked-{ordinal}","D","unlinked-same-name",f"{alias}/{path}",{},
            checkpoint_type(raw),declared,raw,{},"not established","","","not established",
            "actual hash differs from the frozen production initialization")
        hashes[-1].update(sha256_expected=FOUNDATION,verification="computed; differs from frozen foundation")
    with ZipFile(store/"net-path-panel/inference.zip") as bi_infer:
        compact = json.loads(bi_infer.read("models.json"))["weights"]
        grouped = {}
        with tarfile.open(published/"MACE_E0_clip_factorial_archive.tar.gz","r|gz") as z:
            for item in z:
                parts = item.name.split("/")
                if len(parts)==3 and parts[0]=="runs" and item.isfile():
                    if parts[2] in ("model.pt","result.json","training.csv","predictions.csv","path_errors.csv","neb.json"):
                        grouped.setdefault(parts[1],{})[parts[2]]=z.extractfile(item).read()
        for group, files in sorted(grouped.items()):
            marker = json.loads(files["result.json"])
            seed, arm = marker["seed"],marker["arm"]
            expected = marker["output_sha256"]
            if marker["protocol_sha256"]!=digest_from_protocol(published) or marker["steps"]!=config["steps"]:
                raise ValueError(group)
            if any(digest(files[k])!=v for k,v in expected.items()):
                raise ValueError(f"Factorial asset hash mismatch: {group}")
            steps=records(files["training.csv"])
            if len(steps)!=marker["steps"]:
                raise ValueError(f"Factorial training log length mismatch: {group}")
            duplicate = ""
            if arm=="A":
                key=f"D0_{seed}.pt"
                if digest(bi_infer.read(key))!=expected["model.pt"] or compact[key]["sha256"]!=expected["model.pt"]:
                    raise ValueError(key)
                duplicate=f"archives/net-path-panel/inference.zip!{key}"
            add(f"e0-{group}","A",f"factorial-{arm}",f"history/outputs/MACE_E0_clip_factorial_archive.tar.gz!runs/{group}/model.pt",
                marker,checkpoint_type(files["model.pt"]),expected["model.pt"],files["model.pt"],
                dict(config["settings"],clip_norm=marker["treatment"]["clip"]),"278 canonical D0 frames",
                "E0 validation 241 paths / 124 materials" + ("; common panel 107 materials" if arm=="A" else ""),
                f"history/outputs/MACE_E0_clip_factorial_results.json; {NET}/per_material.csv" if arm=="A"
                else "history/outputs/MACE_E0_clip_factorial_results.json",
                f"{len(steps)} steps and prediction/score checksums verified",
                f"E0 clip factorial {marker['treatment']}", "primary seed-spread group" if arm=="A"
                else "separate experimental arm only",marker["order_sha256"],duplicate)

        with ZipFile(published/"MACE_106_cloud_reconstruction/weights_007-018.zip") as z:
            for (arm,seed),job in sorted(bi_jobs.items()):
                prefix = f"v2.0/pair_1/{arm}/seed{seed}/"
                marker = json.loads(z.read(prefix+"result.json"))
                blob = z.read(prefix+"weights.pt")
                expected=marker["files_sha256"]["weights.pt"]
                if digest(bi_infer.read(f"{arm}_{seed}.pt"))!=expected:
                    raise ValueError("Duplicate Bi inference archive disagrees")
                if tuple(marker[k] for k in ("protocol_sha256","indices_sha256","order_sha256")) != tuple(
                    job[k] for k in ("protocol_sha256","frame_indices_sha256","seed_order_sha256")):
                    raise ValueError("Bi provenance mismatch")
                trace = "Step count from frozen run manifest; standalone training trace absent in the weight/prediction packages"
                if prefix+"training.csv" in z.namelist():
                    logs = z.read(prefix+"training.csv")
                    if digest(logs)!=marker["files_sha256"]["training.csv"] or len(records(logs))!=int(job["steps"]):
                        raise ValueError("Bi training log mismatch")
                    trace = f"{len(records(logs))} steps verified"
                add(f"bi-v2-reconstructed-{arm}-{seed}","B","Bi-v2.0-pair1",
                    f"history/outputs/MACE_106_cloud_reconstruction/weights_007-018.zip!{prefix}weights.pt",
                    dict(marker,steps=int(job["steps"])),architecture,expected,blob,info["settings"],
                    f"278 D0 + {job['edge_id']}; extra_xyz_sha256={bi_extra}; indices_sha256={marker['indices_sha256']}",
                    "v2 original 96; reconstruction 106; archived-weight audit 107 materials",
                    f"{NET}/per_material.csv; {NET}/summary.json",trace,
                    "verified reconstruction; Q is a distinct path, not a D0 or replay arm",
                    order=marker["order_sha256"],copies=f"archives/net-path-panel/inference.zip!{arm}_{seed}.pt")

    with tarfile.open(store/"MACE_batch_pilot_GPU_archive.tar","r:") as z:
        for seed in SEEDS:
            p=f"batch_pilot_results/D0/seed{seed}/"
            marker=json.load(z.extractfile(p+"result.json"))
            blob=z.extractfile(p+"weights.pt").read()
            logs=z.extractfile(p+"training.csv").read()
            if digest(logs)!=marker["files"]["training.csv"] or len(records(logs))!=710 or marker["optimizer_steps"]!=710:
                raise ValueError("Batch-pilot D0 training log mismatch")
            if marker["execution_sha256"]!=batch_execution_sha(store):
                raise ValueError("Batch execution mismatch")
            add(f"batch-pilot-D0-{seed}","A","four-path-batch-D0",
                f"archives/MACE_batch_pilot_GPU_archive.tar!{p}weights.pt",marker,architecture,
                marker["files"]["weights.pt"],blob,batch["training"],"278 canonical D0 frames",
                "91 project-held-out oxides","results/four_paths_material_errors.csv",
                "710 steps and training checksum verified","same seed labels; separate 3080 Ti run",
                "separate seed-spread group",marker["order_sha256"])

    groups = (
        (PHASE+"/run_provenance.json","phase1-p0","mp-753143"),
        (PHASE+"/confirmation/run_provenance.json","phase1-p0/confirm","mp-753143"),
        (PHASE+"/transfer/cr_ga_li_o/run_provenance.json","phase1-p0/received_transfers/next_block/feature_gap","mp-768690"),
    )
    for source_name,run_folder,case in groups:
        for marker in json.loads((repo/source_name).read_bytes()):
            arm,seed = marker["arm"],marker["seed"]
            if arm not in ("pure","path3","replay3","path","replay"):
                continue
            folder = store/run_folder/f"seed{seed}"/arm
            final=marker["optimizer_steps"]
            stop_file=folder/f"done_{final}.json"
            if stop_file.exists() and json.loads(stop_file.read_bytes())!=marker:
                raise ValueError("Phase 1 mirrored marker differs")
            for name,expected in marker["files"].items():
                if not name.endswith(".pt"):
                    continue
                step=int(name.split("_")[1].split(".")[0])
                model_path=folder/name
                logs=(folder/f"training_{final}.csv")
                status="marker logs only"
                if logs.is_file():
                    payload=logs.read_bytes()
                    if digest(payload)!=marker["files"][logs.name] or len(records(payload))!=final:
                        raise ValueError("Phase 1 training log mismatch")
                    status=f"{final} steps, verified final training log"
                treatment = "pure 278 frame D0" if arm=="pure" else (
                    f"canonical D0 plus matched D0 replays for {case}" if arm.startswith("replay") else
                    f"canonical D0 plus labelled images of {case}")
                add(f"p0-{case}-{arm}-{seed}-{step}","A" if arm=="pure" else "B","phase1-protected-schedule",
                    f"archives/{run_folder}/seed{seed}/{arm}/{name}",dict(marker,steps=step),architecture,expected,
                    model_path.read_bytes() if model_path.is_file() else None,phase1["training"],
                    treatment+f"; bundle={marker['bundle']}; delta={marker.get('delta','')}","107 and 26 project-held-out materials",
                    f"{source_name}; {PHASE}/candidate_comparison/paired_materials.csv",
                    status,f"archive fit at {final} steps; model checkpoint at {step}",
                    "bytes required before consideration" if arm=="pure" else "excluded",marker["schedule"])

    report_text = audit_prediction_keys(repo,store)
    args.output.mkdir(parents=True,exist_ok=False)
    write_table(args.output,"checkpoint_inventory.csv",records_by_id)
    write_table(args.output,"checkpoint_provenance.csv",provenance)
    write_table(args.output,"checkpoint_hashes.csv",hashes)
    print("Audited",len(records_by_id),"logical checkpoints",Counter(
        (r["class_id"],r["availability"]) for r in records_by_id))
    print(report_text)
    print("D0 SHA256",data_identity)


def digest_from_protocol(published):
    with ZipFile(published/"MACE_E0_clip_factorial_v1.zip") as z:
        return z.read("protocol.sha256").decode().strip().split()[0]


def batch_execution_sha(store):
    with ZipFile(store/"MACE_batch_pilot_GPU_bundle.zip") as z:
        return digest(json.dumps(json.loads(z.read("execution.json")),sort_keys=True).encode())


def audit_prediction_keys(repo, store):
    with ZipFile(store/"net-path-panel/predictions.zip") as z:
        vectors=records(z.read("predictions.csv"))
    metrics=records((repo/NET/"per_material.csv").read_bytes())
    groups = {}
    for arm in ("D0","P","Q"):
        for seed in SEEDS:
            images=[(r["edge_id"],r["material_id"],r["frame"]) for r in vectors
                    if r["arm"]==arm and r["seed"]==str(seed)]
            scores=[(r["edge_id"],r["material_id"]) for r in metrics
                    if r["arm"]==arm and r["seed"]==str(seed)]
            if len(images)!=595 or len(set(images))!=595 or len(scores)!=107 or len(set(scores))!=107:
                raise ValueError(f"Incomplete or duplicated prediction keys: {arm} {seed}")
            if set(scores)!={image[:2] for image in images}:
                raise ValueError(f"Material and image keys differ: {arm} {seed}")
            groups[arm,seed]=set(images)
    if any(keys!=groups["D0",17] for keys in groups.values()):
        raise ValueError("Prediction arms and seeds use different image keys")
    return f"Verified {len(vectors)} archived image energies on common keys; 321 D0 material-seed scores."


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","archives","history","project","output"):
        p.add_argument(name,type=Path)
    audit(p.parse_args())
