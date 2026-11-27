"""Reconcile recorded QE jobs and inspect public LiTraj training data."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import struct
from zipfile import ZipFile
import zlib

from ase.io import read as ase_read
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def qe_costs(results):
    old = results / "project_data_audit_20261127/dft_cost_ledger.csv"
    extra = results / "phase_b4_r2/cutoff_control/overnight_compute_ledger.csv"
    slots = results / "phase_b4_r2_preflight/h1c_cost_inputs.csv"
    snapshot = results / "mp1191903_ci_pilot/first_acceptance_20261127_2000.json"
    jobs = []
    for row in rows(old):
        if not (results / row["source"]).is_file():
            raise ValueError(f"Missing cost source: {row['source']}")
        jobs.append({k: (float(v) if v else None) if k.endswith("_s") else v
                     for k, v in row.items()})
    for row in rows(extra):
        seconds = int(row["solver_seconds"])
        jobs.append(dict(material="mp-1191903", phase="fixed_geometry_precision", role=row["job"],
                         hardware="RTX 3090", grid=row["mesh"],
                         cutoff_Ry=f"{row['wavefunction_cutoff_Ry']}/{row['density_cutoff_Ry']}",
                         solver_wall_s=seconds, gpu_occupied_s=seconds, assigned_rank_s=None,
                         outcome="complete", source=str(extra.relative_to(results).as_posix()),
                         start_utc=row["start_utc"]))
    identities = {(row["source"], row["material"], row["phase"], row["role"], row["hardware"], row["grid"])
                  for row in jobs}
    if len(identities) != len(jobs):
        raise ValueError("Duplicate cost jobs")

    totals = {}
    for name in sorted({row["hardware"] for row in jobs}):
        subset = [row for row in jobs if row["hardware"] == name]
        sums = {}
        for key in ("solver_wall_s", "gpu_occupied_s", "assigned_rank_s"):
            values = [row[key] for row in subset if row[key] is not None]
            sums[key] = sum(values) if values else None
        totals[name] = dict(events=len(subset), **sums)
    scf = [row["solver_wall_s"] for row in jobs if row["material"] == "mp-1191903"
           and row["hardware"] == "RTX 3090" and row["grid"] == "2x2x2"
           and row["cutoff_Ry"] == "50/400"]
    pool = rows(slots)
    previous = json.loads(snapshot.read_bytes())
    return dict(
        sources={str(p.relative_to(results).as_posix()): digest(p)
                 for p in (old, extra, slots, snapshot)},
        jobs=jobs, archived_totals_by_hardware=totals,
        matched_fixed_geometry_solver_seconds=dict(n=len(scf), minimum=min(scf), maximum=max(scf)),
        candidate_cost_input=dict(
            paths=len(pool), atom_count_min=min(int(r["atoms"]) for r in pool),
            atom_count_max=max(int(r["atoms"]) for r in pool),
            proxy_kpoint_grid_counts=dict(Counter(r["gamma_centered_mesh_delta_0_25"] for r in pool)),
            populated_estimate_fields=sum(bool(r["estimated_total_cost"]) for r in pool),
        ),
        qe_a_archived_status=dict(
            snapshot_at=previous["observed_at"], endpoint_a_converged=previous["endpoint_a"]["validated"],
            endpoint_b_started=previous["endpoint_b"]["started"],
            path_probe=previous["path_probe"], complete_measured_path_cost=None,
            runtime_sums_included=False,
        ),
        totals_scope="Archived solver events only; failed and partial events are counted once.",
    )


def frame_data(text):
    return ase_read(io.StringIO(text), index=":", format="extxyz")


def inspect_labels(atoms):
    if atoms.calc is None:
        raise ValueError("Missing calculator labels")
    actual = atoms.calc.results
    for name in ("energy", "forces", "stress"):
        if not np.isfinite(actual[name]).all():
            raise ValueError(f"Nonfinite {name}")
    if actual["forces"].shape != (len(atoms), 3):
        raise ValueError("Forces are not mapped to every atom")
    return set(actual)


ENERGY = re.compile(rb"(?:^|\s)energy=([-+0-9.eE]+)(?:\s|$)")


def archived_energies(archive, name):
    values = []
    with archive.open(name) as stream:
        while ln := stream.readline():
            size = int(ln)
            header = ENERGY.search(stream.readline())
            if not header:
                raise ValueError(f"Missing frame energy: {name}")
            values.append(float(header[1]))
            for _ in range(size):
                if not stream.readline():
                    raise ValueError(f"Incomplete frame: {name}")
    return np.asarray(values)


def neb_archive(path, protected_base, h1_candidates):
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        index = "nebDFT2k/nebDFT2k_index.csv"
        cohort = list(csv.DictReader(archive.read(index).decode("utf-8-sig").splitlines()))
        identifiers = {row["edge_id"] for row in cohort}
        parents = {row["material_id"] for row in cohort}
        if len(cohort) != len(identifiers):
            raise ValueError("Repeated nebDFT2k path IDs")
        initial = {r["edge_id"] for r in cohort if f"nebDFT2k/{r['edge_id']}_init.xyz" in names}
        relaxed = {r["edge_id"] for r in cohort if f"nebDFT2k/{r['edge_id']}_relaxed.xyz" in names}
        train = sorted([row for row in cohort if row["_split"] == "train"], key=lambda r: r["edge_id"])
        offsets = defaultdict(list)
        for row in train:
            e = archived_energies(archive, f"nebDFT2k/{row['edge_id']}_relaxed.xyz")
            offsets["forward_first_image"].append(float(e.max() - e[0] - float(row["em_dft"])))
            offsets["lower_endpoint"].append(float(e.max() - min(e[0], e[-1]) - float(row["em_dft"])))
            offsets["full_profile_range"].append(float(e.max() - e.min() - float(row["em_dft"])))
        sample = []
        for i in np.linspace(0, len(train) - 1, 6, dtype=int):
            row = train[i]
            imgs = frame_data(archive.read(f"nebDFT2k/{row['edge_id']}_relaxed.xyz").decode())
            label_keys = set.intersection(*(inspect_labels(atoms) for atoms in imgs))
            sample.append(dict(edge_id=row["edge_id"], images=len(imgs), label_keys=sorted(label_keys)))
    with ZipFile(protected_base) as archive:
        training_table = archive.read("training_frames.csv")
        expected = json.loads(archive.read("plan.json"))["sha256"]["training_frames.csv"]
        if sha256(training_table).hexdigest() != expected:
            raise ValueError("D0 frame metadata differs from frozen bundle")
        d0 = list(csv.DictReader(training_table.decode().splitlines()))[:278]
    h1 = rows(h1_candidates)
    d0_edges = {row["edge_id"] for row in d0}
    d0_parents = {row["material_id"] for row in d0}
    return dict(
        archive_sha256=digest(path), path_count=len(identifiers), material_count=len(parents),
        per_split_paths=dict(Counter(row["_split"] for row in cohort)),
        path_pairs_complete=len(initial & relaxed), training_frame_samples=sample,
        qe_a_exact_material_present="mp-1191903" in parents,
        h1_exact_candidate_edges_in_archive=sum(row["edge_id"] in identifiers for row in h1),
        h1_exact_candidate_materials_in_archive=len({row["material_id"] for row in h1} & parents),
        matching_D0_exact_edges=len(d0_edges & identifiers),
        matching_D0_exact_materials=len(d0_parents & parents),
        training_barrier_definition_audit={
            key: dict(within_1_meV=int(np.count_nonzero(np.abs(values) <= .001)),
                      max_absolute_difference_meV=float(np.max(np.abs(values)) * 1000))
            for key, values in offsets.items()
        },
        training_paths_with_reconstructed_barrier=len(train),
        scope="Index and member names only for validation and test; label samples from official train.",
    ), cohort


def first_training_member(raw):
    offset = 0
    local = struct.Struct("<4s5H3I2H")
    while offset + local.size <= len(raw):
        sig, _, flags, compression, _, _, crc, packed_size, size, name_len, extra_len = local.unpack_from(raw, offset)
        if sig != b"PK\x03\x04":
            raise ValueError("Unexpected ZIP header in prefix")
        start = offset + local.size
        member = raw[start:start + name_len].decode()
        data_start = start + name_len + extra_len
        if member.endswith("_train.xyz") and compression == 8 and not flags & 9:
            return member, data_start, packed_size, size, crc
        if not member.endswith("/") or flags & 8:
            raise ValueError("Train member must directly follow directory entries")
        offset = data_start + packed_size
    raise ValueError("No train member in prefix")


def mpl_prefix(path, neb_rows):
    raw = path.read_bytes()
    member, start, packed_size, _, _ = first_training_member(raw)
    text = zlib.decompressobj(-15).decompress(raw[start:start+packed_size], 24 * 1024 * 1024)
    lines = text.splitlines(keepends=True)
    position, end, frames = 0, 0, []
    while position < len(lines) and lines[position].strip():
        size = int(lines[position])
        end = position + size + 2
        if end > len(lines) or not lines[end - 1].endswith(b"\n"):
            break
        frames.append(b"".join(lines[position:end]))
        position = end
    if len(frames) < 10:
        raise ValueError("Too few complete public training frames")
    all_frames = frame_data(b"".join(frames).decode())
    info_counts, labels, materials = Counter(), Counter(), Counter()
    for atoms in all_frames:
        info_counts.update(atoms.info.keys())
        labels.update(inspect_labels(atoms))
        materials[atoms.info["material_id"]] += 1
    edges_per_material = Counter(row["material_id"] for row in neb_rows)
    return dict(
        prefix_sha256=digest(path), compressed_prefix_bytes=path.stat().st_size, member=member,
        complete_train_frames=len(all_frames), material_count=len(materials),
        material_id_histogram=materials, metadata_fields_by_frame=info_counts, labels_by_frame=labels,
        material_ids_linked_to_nebDFT2k=len(materials.keys() & edges_per_material.keys()),
        linked_materials_with_multiple_nebDFT2k_paths=sum(edges_per_material[key] > 1 for key in materials),
        count_of_nebDFT2k_paths_by_matching_material={key: edges_per_material[key] for key in materials},
        has_path_id_in_sample=any(field in info_counts for field in ("edge_id", "path_id", "trajectory_id")),
        has_iteration_in_sample=any(field in info_counts for field in ("step", "iteration", "ionic_step")),
        has_job_timing_in_sample=any(field in info_counts for field in ("walltime", "elapsed_seconds", "cpu_time")),
        scope="Contiguous prefix of official subsample train member; no inference about unsampled fields.",
    )


def audit_train_member(chunks, neb_rows):
    with chunks[0].open("rb") as first:
        member, start, packed_size, size, expected_crc = first_training_member(first.read(1024))
    if sum(path.stat().st_size for path in chunks) != start + packed_size:
        raise ValueError("Compressed parts do not span exactly the training member")
    info = Counter()
    materials = Counter()
    label_fields = Counter()
    digest_stream = sha256()
    decompressor = zlib.decompressobj(-15)
    carry = b""
    body_left = 0
    header_next = False
    frames = raw_size = crc = 0
    for i, path in enumerate(chunks):
        with path.open("rb") as handle:
            if i == 0:
                handle.seek(start)
            while compressed := handle.read(512 * 1024):
                data = decompressor.decompress(compressed)
                raw_size += len(data)
                crc = zlib.crc32(data, crc)
                digest_stream.update(data)
                lines = (carry + data).split(b"\n")
                carry = lines.pop()
                for line in lines:
                    if header_next:
                        fields = set(re.findall(rb"(?<!\S)(\w+)=", line))
                        info.update(key.decode() for key in fields)
                        for key in ("energy", "stress"):
                            label_fields[key] += key.encode() in fields
                        label_fields["forces"] += b":forces:R:3" in line
                        material = re.search(rb"(?<!\S)material_id=(mp-[0-9]+)(?:\s|$)", line)
                        materials[material[1].decode() if material else "missing"] += 1
                        header_next = False
                        frames += 1
                    elif body_left:
                        body_left -= 1
                    else:
                        body_left = int(line)
                        header_next = True
    if not decompressor.eof or (raw_size, crc) != (size, expected_crc) or carry or body_left or header_next:
        raise ValueError("Incomplete or invalid compressed training stream")
    counts = Counter(row["material_id"] for row in neb_rows)
    return dict(member=member, archive_crc32=f"{crc:08x}", decoded_sha256=digest_stream.hexdigest(),
                complete_train_frames=frames, distinct_materials=len(materials),
                same_material_in_nebDFT2k=len(materials.keys() & counts.keys()),
                shared_materials_with_multiple_nebDFT2k_paths=sum(counts[key] > 1 for key in materials),
                header_fields_by_frame=dict(info), labels_present_by_frame=dict(label_fields),
                scope="All subsample train headers; finite atomic labels checked on bounded sample only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("results", "neb", "d0_bundle", "mpl_train_prefix", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--mpl-train-part", action="append", type=Path, default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    costs = qe_costs(args.results)
    neb, index = neb_archive(args.neb, args.d0_bundle,
                             args.results / "phase_h1/structure_output/candidate_ranking_and_audit.csv")
    data = dict(nebDFT2k=neb, MPLiTrj_subsample_train_prefix=mpl_prefix(args.mpl_train_prefix, index),
                script_sha256=digest(Path(__file__)))
    if args.mpl_train_part:
        data["MPLiTrj_subsample_train_full"] = audit_train_member(args.mpl_train_part, index)
    save(args.output / "recorded_costs.json", costs)
    save(args.output / "public_data_reuse.json", data)


if __name__ == "__main__":
    main()
