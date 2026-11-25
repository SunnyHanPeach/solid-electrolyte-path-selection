"""Extend an existing B2 path batch without changing earlier orders or selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from audit_phase_b0 import frame_prefix
from freeze_b2_overnight_01 import make_orders, write
from freeze_net_path_panel import csv_bytes
from phase_a0_geometry import atoms_from_bytes, parse_path
from phase_a1_descriptors import load_representatives, radial_descriptors
from run_b2_overnight_01 import assets
from audit_signed_barriers import file_sha, rows, sha

EXTRA_SEEDS = (53, 67)


def geometry_novelty(args, d0, candidates):
    bank_dir = args.repo/"results/phase_a01"
    centers = sorted(load_representatives(bank_dir/"reference_material_distribution.csv", d0))
    ref = []
    for frame, li in centers:
        atom = d0[frame-1]
        ref.append(radial_descriptors(atom, np.array([li-1]), sorted(set(atom.get_chemical_symbols())))["rdf_geometry"][0])
    ref = np.array(ref)
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-15)
    out = {}
    with ZipFile(args.neb) as z:
        for edge, row in candidates.items():
            li = int(row["moving_Li_row"])-1
            _, _, atoms = parse_path(z, edge, "init", li)
            desc = np.array([radial_descriptors(a, np.array([li]), sorted(set(a.get_chemical_symbols())))["rdf_geometry"][0]
                             for a in atoms])
            desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-15)
            value = float(np.maximum(0., 1. - (desc @ ref.T).max(axis=1)).mean())
            out[edge] = value
    return out


def extend(args):
    import sys
    sys.path.insert(0, str(args.project/"experiment2"))
    from prepare_data import labeled_frame, metadata, raw_frames

    six, base, bundle, frame_rows, orders = assets(args)
    if file_sha(args.neb) != six["reference_archive_sha256"] or file_sha(args.features) != six["features_sha256"]:
        raise ValueError("Changed input source")
    with ZipFile(args.base) as z:
        d0 = atoms_from_bytes(frame_prefix(z.read("training.xyz"), 278))
        old_rows = rows(z.read("training_frames.csv"))
    report = args.repo/"results/b2_overnight_01"
    quality = {r["edge_id"]: r for r in rows((report/"candidate_integrity.csv").read_bytes())}
    meta = {r["edge_id"]: r for r in rows(args.features.read_bytes()) if r["edge_id"] in quality}
    selected = [r["edge_id"] for r in six["candidates"]]
    options = [e for e in sorted(meta) if not quality[e]["eligibility_reason"]]
    novelty = geometry_novelty(args, d0, {e:meta[e] for e in sorted(set(options) | set(selected))})
    q_novelty = np.quantile([novelty[e] for e in options], [1/3, 2/3])
    novelty_stratum = {e: int(np.searchsorted(q_novelty, v, side="right")) for e,v in novelty.items()}
    groups = (("coordination_start", "coordination_end", "coordination_range", "min_Li_anion_distance_A"),
              ("hop_displacement_A", "path_excess_length_A", "total_turn_deg"),
              ("path_length_A",), ("rdf_novelty",))
    columns = [c for group in groups for c in group]

    def values(edge):
        return np.array([novelty[edge] if c=="rdf_novelty" else float(meta[edge][c]) for c in columns])

    q1, q3 = np.quantile([values(e) for e in options], [.25, .75], axis=0)
    scale = np.where(q3 > q1, q3-q1, 1.)
    group_scale = np.repeat(np.sqrt([len(g) for g in groups]), [len(g) for g in groups])
    starting_count = len(selected)
    if args.target_paths <= starting_count:
        raise ValueError("Requested path count must exceed the earlier freeze")
    previous_prefix = "" if starting_count == 6 else "revised_"
    prefix = "revised_" if starting_count == 6 else "six_hour_"
    prior = rows((report/f"{previous_prefix}candidate_ranking.csv").read_bytes())
    extended = []
    for i in range(starting_count - 2, args.target_paths - 2):
        occupied = {meta[e]["chemsys"] for e in selected}
        materials = {meta[e]["material_id"] for e in selected}
        anions = {meta[e]["anion_family"] for e in selected}
        novelty_population = Counter(novelty_stratum[e] for e in selected)
        eligible, excluded = [], []
        other = np.array([values(e) for e in selected])
        for e,m in sorted(meta.items()):
            reason = quality[e]["eligibility_reason"]
            if not reason and (m["chemsys"] in occupied or m["material_id"] in materials or e in selected):
                reason = "same material or exact chemistry as a locked path"
            row = dict(round=i, edge_id=e, material_id=m["material_id"], chemsys=m["chemsys"],
                       anion_family=m["anion_family"], atoms=m["atoms_per_image"], images=m["init_images"],
                       **{k:m[k] for k in six["feature_iqr_scale"]},
                       rank="", selected=0, excluded_reason=reason, new_anion="", min_feature_distance="",
                       path_length_A=m["path_length_A"], rdf_novelty=novelty.get(e,""),
                       rdf_novelty_stratum=novelty_stratum.get(e,""))
            if reason:
                excluded.append(row)
            else:
                row["new_anion"] = int(m["anion_family"] not in anions)
                row["min_feature_distance"] = float(np.linalg.norm(
                    (other - values(e))/(scale*group_scale), axis=1).min())
                eligible.append(row)
        eligible.sort(key=lambda r: (-r["new_anion"], novelty_population[r["rdf_novelty_stratum"]],
                                     -r["min_feature_distance"], r["edge_id"]))
        if not eligible:
            raise ValueError("Insufficient valid uncovered chemistries")
        eligible[0]["selected"] = 1
        for rank, r in enumerate(eligible, 1):
            r["rank"] = rank
        selected.append(eligible[0]["edge_id"])
        extended.extend(eligible + excluded)

    added = bytearray(bundle["added.xyz"])
    frame_sizes = [int(r["n_atoms"]) for r in frame_rows]
    added_rows = rows(bundle["added_frames.csv"])
    paths = list(six["candidates"])
    with ZipFile(args.neb) as z:
        for slot, edge in enumerate(selected[starting_count:], starting_count + 1):
            row = meta[edge]
            member = f"nebDFT2k/{edge}_relaxed.xyz"
            blob = z.read(member)
            evidence = quality[edge].copy()
            assert sha(blob) == evidence["relaxed_sha256"]
            path_frames = [dict(edge_id=edge, material_id=row["material_id"], frame=i, role="candidate",
                                n_atoms=len(body), energy_eV=labeled_frame(metadata(head), body)[0])
                           for i,_,_,head,body in raw_frames(z,member)]
            ids = list(range(len(frame_sizes), len(frame_sizes)+len(path_frames)))
            frame_sizes.extend(p["n_atoms"] for p in path_frames)
            for seed in ("17", "29", "43"):
                job = orders[seed]
                job[f"path{slot}"], job[f"replay{slot}"] = make_orders(job["pure"], ids, frame_sizes, int(seed))
            added.extend(blob)
            added_rows.extend(path_frames)
            evidence.pop("edge_id")
            evidence.pop("eligibility_reason")
            paths.append(dict(slot=slot, cohort="new_matched_label", edge_id=edge, material_id=row["material_id"],
                              chemsys=row["chemsys"], anion_family=row["anion_family"], images=len(ids),
                              atoms=row["atoms_per_image"], optimizer_steps=710+5*len(ids),
                              path_presentations_per_image=5, selected_ids=ids,
                              prior_comparability="No prior identical protected-schedule experiment selected", **evidence))

    confirmation = json.loads(args.confirmation.read_bytes())
    if confirmation["bundle"] != six["base_bundle_sha256"] or confirmation["path"] != selected[0]:
        raise ValueError("Wrong Phase 1 Seed 53/67 confirmation")
    old_baselines = json.loads(bundle["baselines.json"])
    for seed in EXTRA_SEEDS:
        s = str(seed)
        original = confirmation["orders"][s]
        path, replay = make_orders(original["pure"], paths[0]["selected_ids"], frame_sizes, seed)
        for key, order in (("path3", path), ("replay3", replay)):
            a = [[(old_rows[j]["edge_id"], str(old_rows[j]["frame"])) for j in b] for b in original[key]]
            canonical = old_rows[:278] + added_rows
            b = [[(canonical[j]["edge_id"], str(canonical[j]["frame"])) for j in batch] for batch in order]
            if a != b:
                raise ValueError("Extra seed changed the protected schedules")
        marker = json.loads((args.confirm_baseline/f"seed{seed}/pure/done_710.json").read_bytes())
        if (marker["bundle"], marker["confirmation"], marker["schedule"]) != (
                six["base_bundle_sha256"], file_sha(args.confirmation), sha(json.dumps(original["pure"]).encode())):
            raise ValueError("Pure D0 extra-seed identity changed")
        old_baselines[s] = marker
        orders[s] = dict(pure=original["pure"], path1=path, replay1=replay)

    for r in prior:
        e = r["edge_id"]
        r.update(path_length_A=meta[e]["path_length_A"], rdf_novelty=novelty.get(e,""),
                 rdf_novelty_stratum=novelty_stratum.get(e,""))
    new_data = dict(bundle, **{"added.xyz":bytes(added), "added_frames.csv":csv_bytes(added_rows),
                              "orders.json":json.dumps(orders,separators=(",",":")).encode(),
                              "baselines.json":json.dumps(old_baselines).encode()})
    previous_sha = file_sha(args.matrix)
    earlier_maps = dict(six.get("compatible_matrix_sha256_slots", {}), **{previous_sha:starting_count})
    updated = dict(six, id="B2-Overnight-01-" + prefix.rstrip("_"), planned_fits=6*args.target_paths+4,
                   candidates=paths, extra_seeds=EXTRA_SEEDS,
                   earlier_matrix_sha256=six.get("earlier_matrix_sha256",previous_sha),
                   compatible_matrix_sha256_slots=earlier_maps, confirmation_sha256=file_sha(args.confirmation),
                   revised_groups=groups, rdf_reference_sites=2469, rdf_tertiles=q_novelty.tolist(),
                   revised_iqr_scale=dict(zip(columns, scale.tolist())),
                   revised_selection_rule=("Keep the frozen slots. For added slots, prefer uncovered "
                                           "anion, then the least represented RDF novelty tertile, then "
                                           "maximin over scaled coordination, geometry, arc length and novelty."),
                   sha256={k:sha(v) for k,v in new_data.items()})
    queue = []
    for slot in range(1,args.target_paths+1):
        for s in (17,29,43):
            for a in ("path","replay"):
                queue.append((slot,s,f"{a}{slot}"))
    queue.extend((1,s,a) for s in EXTRA_SEEDS for a in ("path1","replay1"))
    assert len(queue)==updated["planned_fits"]
    write(report/f"{prefix}plan.json", (json.dumps(updated,indent=2,ensure_ascii=False)+"\n").encode())
    write(report/f"{prefix}candidates.csv", csv_bytes([dict(p, path_length_A=meta[p["edge_id"]]["path_length_A"],
                    rdf_novelty=novelty[p["edge_id"]], rdf_novelty_stratum=novelty_stratum[p["edge_id"]]) for p in paths]))
    write(report/f"{prefix}candidate_ranking.csv", csv_bytes(prior+extended))
    write(report/f"{prefix}tasks.csv", csv_bytes([dict(order=i,slot=slot,edge_id=paths[slot-1]["edge_id"],
          seed=s,arm=a,steps=len(orders[str(s)][a]),order_sha256=sha(json.dumps(orders[str(s)][a]).encode()))
          for i,(slot,s,a) in enumerate(queue,1)]))
    with ZipFile(args.output,"x",compression=ZIP_DEFLATED) as z:
        for k,v in new_data.items():
            z.writestr(k,v)
        z.write(report/f"{prefix}plan.json", "plan.json")
    print([(p["slot"],p["edge_id"],p["chemsys"]) for p in paths], flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","project","base","matrix","neb","features","confirmation","confirm_baseline","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--target-paths",type=int,default=10)
    extend(parser.parse_args())
