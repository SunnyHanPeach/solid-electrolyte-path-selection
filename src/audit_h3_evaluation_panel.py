"""Audit QE reference availability and local overlap of the frozen H2 geometry panel."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import md5, sha256
from itertools import combinations, product
import json
from pathlib import Path
import re
import tarfile

from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list
import numpy as np
from pymatgen.core import Structure

from audit_phase_h2_evaluation import digest, rows
from audit_public_neb_panel import geometry_hash, groups
from audit_public_neb_reuse import FP_HASH, checked
from phase_a0_geometry import reaction_coordinate, signature_from_vectors
from phase_h1_shortlist import write
from prepare_h2_eval_geometry_panel import save_json
from prepare_phase_b4_spin_pilot import SSSP_METADATA_MD5, attributes, numeric


RADII = (2.8, 3.0, 3.2)
TOLERANCES = (.05, .10, .15)
BASELINE_RADIUS = 3.0
BASELINE_TOLERANCE = .10
SSSP_ARCHIVE_MD5 = "a58f1b3373f330179fd0832c48bb9a52"
SPIN_REVIEW_ELEMENTS = set("""
Sc Ti V Cr Mn Fe Co Ni Cu Y Zr Nb Mo Tc Ru Rh Pd Ag Hf Ta W Re Os Ir Pt Au
Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb U Np Pu Am
""".split())


def upf_header(raw, element):
    body = raw.decode("utf-8", errors="replace")
    tag = re.search(r"<PP_HEADER\b([^>]*)>", body, re.S)
    if tag is None:
        raise ValueError("Missing PP_HEADER")
    header = attributes(tag[1])
    if header:
        symbol = header["element"].strip()
        xc = header["functional"].strip()
        z = numeric(header["z_valence"])
        relativity = header.get("relativistic", "").strip()
        has_so = header.get("has_so", "").strip().lower()
        so = {"t": True, "true": True, "f": False, "false": False}.get(has_so)
        pseudo_type = header["pseudo_type"].strip()
        cutoffs = [numeric(header.get(key, "0")) for key in ("wfc_cutoff", "rho_cutoff")]
        orbitals = [attributes(s) for s in re.findall(r"<PP_CHI\.\d+\b([^>]*)>", body, re.S)]
        occupations = {r["label"]: numeric(r["occupation"]) for r in orbitals
                       if "label" in r and "occupation" in r}
        version_tag = re.search(r"<UPF\s+version=[\"']([^\"']+)", body)
        upf_version = version_tag[1] if version_tag else ""
    else:
        legacy = body[tag.end():body.index("</PP_HEADER>", tag.end())]
        lines = [line.strip() for line in legacy.splitlines() if line.strip()]
        symbol = next(line.split()[0] for line in lines if re.fullmatch(r"[A-Z][a-z]?\s+Element", line))
        xc = next(line.split("Exchange-Correlation", 1)[0].strip()
                  for line in lines if "Exchange-Correlation" in line)
        z = numeric(next(line.split()[0] for line in lines if "Z valence" in line))
        relativity = ("nonrelativistic" if "Non-Relativistic Calculation" in body
                      else "scalar" if "Scalar-Relativistic Calculation" in body else "")
        so = False if relativity else None
        pseudo_type = next((line.split()[0] for line in lines if "pseudopotential" in line), "")
        cutoffs = [numeric(x) for x in next(line for line in lines
                                          if "Suggested cutoff" in line).split()[:2]]
        occupations = {tokens[0]: numeric(tokens[-1]) for line in lines
                       if len(tokens := line.split()) == 3 and re.fullmatch(r"\d+[SPDF]", tokens[0])}
        upf_version = "1"
    codes = set(xc.upper().split())
    if symbol != element or not ("PBE" in codes or {"PBX", "PBC"} <= codes):
        raise ValueError(f"Unexpected UPF element or exchange correlation: {symbol}, {xc}")
    if not np.isfinite(z) or z <= 0 or not z.is_integer():
        raise ValueError("Integer pseudopotential valence could not be verified")
    return dict(upf_version=upf_version, functional_header=xc, z_valence=int(z),
                relativistic=relativity, spin_orbit=so, pseudo_type=pseudo_type,
                header_cutoffs_Ry=cutoffs, orbital_occupations=occupations)


def pseudo_inventory(required, catalogue_file, directories, archives):
    catalogue_data = catalogue_file.read_bytes()
    if md5(catalogue_data).hexdigest() != SSSP_METADATA_MD5:
        raise ValueError("SSSP metadata digest differs")
    catalogue = json.loads(catalogue_data)
    names = {catalogue[e]["filename"]: e for e in required if e in catalogue}
    records = {e: dict(element=e, catalogue=catalogue.get(e), file_status="catalogue_only"
                       if e in catalogue else "no_catalogue_entry") for e in sorted(required)}
    archive_status = []

    def inspect(name, raw, origin):
        element = names[name]
        row = records[element]
        if row["file_status"] == "verified_pbe_header":
            return
        if md5(raw).hexdigest() != row["catalogue"]["md5"]:
            row["checksum_mismatch_seen"] = True
            return
        row.update(file_status="verified_bytes_header_pending",
                   file_sha256=sha256(raw).hexdigest(), source=origin)
        try:
            row.update(upf_header(raw, element))
        except (ValueError, KeyError, StopIteration) as exc:
            row["header_issue"] = f"{type(exc).__name__}: {exc}"
        else:
            row["file_status"] = "verified_pbe_header"

    for directory in directories:
        for name in names:
            path = directory / name
            if path.is_file():
                inspect(name, path.read_bytes(), f"{directory.name}/{name}")
    for archive in archives:
        if not archive.is_file():
            continue
        full_md5 = md5(archive.read_bytes()).hexdigest() == SSSP_ARCHIVE_MD5
        files = 0
        try:
            with tarfile.open(archive, "r|gz") as packet:
                for member in packet:
                    name = Path(member.name).name
                    if member.isfile() and name in names:
                        inspect(name, packet.extractfile(member).read(), f"{archive.name}/{name}")
                        files += 1
        except (EOFError, tarfile.TarError):
            full_md5 = False
        archive_status.append(dict(file=archive.name, official_full_archive_md5_match=full_md5,
                                   required_members_read=files))
    return records, archive_status


def qe_path(row, atoms, records):
    counts = Counter(atoms.get_chemical_symbols())
    if set(counts) != set(row["chemsys"].split("-")):
        raise ValueError(f"Panel species differ from seed in {row['slot']}")
    unknown_valences = sorted(e for e in counts if "z_valence" not in records[e])
    zsum = sum(counts[e] * records[e]["z_valence"] for e in counts) if not unknown_valences else None
    spin_flags = sorted(set(counts).intersection(SPIN_REVIEW_ELEMENTS))
    header_pending = sorted(e for e in counts if records[e]["file_status"] != "verified_pbe_header")
    cutoffs = [max(max(records[e]["header_cutoffs_Ry"][i], records[e]["catalogue"][key])
                   for e in counts) if not header_pending else None
               for i, key in enumerate(("cutoff_wfc", "cutoff_rho"))]
    return dict(slot=row["slot"], role=row["role"], stratum=row["stratum"], atoms=len(atoms),
                stoichiometry=dict(counts), pending_upf=header_pending,
                neutral_all_electron_count=int(sum(atoms.numbers)),
                neutral_valence_electron_count=zsum,
                neutral_odd_valence=None if zsum is None else bool(zsum % 2),
                magnetic_element_screen=spin_flags,
                spin_orbit_upf=sorted(e for e in counts if records[e].get("spin_orbit") is True),
                spin_orbit_header_unresolved=sorted(e for e in counts if "z_valence" in records[e]
                                                   and records[e]["spin_orbit"] is None),
                cutoff_reference_wfc_Ry=cutoffs[0], cutoff_reference_rho_Ry=cutoffs[1],
                charge_and_spin_protocol="pending", production_DFT_qualified=False)


def references(result, h1_geometry, fp_reference, unmatched):
    expected = json.loads((result / "phase_b5_r/manifest.json").read_bytes())
    checked(h1_geometry, expected["geometry_sha256"]["candidate_images.xyz"])
    h1 = {row["edge_id"] for row in rows(result / "phase_h1/structure_output/candidate_ranking_and_audit.csv")}
    h1_frames = defaultdict(list)
    for frame in read(h1_geometry, ":"):
        if frame.info["edge_id"] in h1:
            h1_frames[frame.info["edge_id"]].append(frame)
    if len(h1_frames) != 24:
        raise ValueError("The comparison does not contain the 24 frozen H1 paths")
    for key, frames in sorted(h1_frames.items()):
        frames.sort(key=lambda a: a.info["image"])
        mover = int(frames[0].info["moving"])
        if any(int(a.info["moving"]) != mover for a in frames):
            raise ValueError(f"H1 migrant changes in {key}")
        yield key, "H1", frames, mover

    panel = rows(result / "public_neb_panel/panel_paths.csv")
    source = json.loads(gzip.decompress(checked(fp_reference, FP_HASH)))["pathways"]
    if len(panel) != 62:
        raise ValueError("The comparison does not contain the 62 frozen FP-DeErr paths")
    for row in panel:
        key = row["path_id"]
        images = source[key]["dft_neb_reference"]["images"]
        if geometry_hash(images) != row["ordered_DFT_geometry_sha256"]:
            raise ValueError(f"FP-DeErr geometry changed for {key}")
        frames = []
        for image in images:
            s = Structure.from_dict(image["structure"])
            frames.append(Atoms([site.specie.symbol for site in s], positions=s.cart_coords,
                                cell=s.lattice.matrix, pbc=True))
        _, delta = find_mic(frames[-1].positions - frames[0].positions, frames[0].cell, True)
        lithium = np.flatnonzero(frames[0].numbers == 3)
        mover = int(lithium[np.argmax(delta[lithium])])
        if delta[mover] < .75 or delta[mover] < 2 * np.max(np.delete(delta, mover)):
            unmatched.append(dict(path=key, reason="relaxed geometric mover not isolated"))
            continue
        yield key, "FP-DeErr_relaxed_geometry", frames, mover


def local_features(entry):
    key, cohort, frames, mover = entry
    first = frames[0]
    if first.numbers[mover] != 3 or any(not np.array_equal(a.numbers, first.numbers)
                                        or not np.allclose(a.cell, first.cell, atol=1e-6, rtol=0)
                                        or not a.pbc.all() for a in frames):
        raise ValueError(f"Atom or cell mapping differs within path {key}")
    _, _, coordinates = reaction_coordinate([a.positions[mover] for a in frames], first.cell, first.pbc)
    middle = 1 + int(np.argmin(np.abs(coordinates[1:-1] - .5)))
    sigs, image_rows = defaultdict(list), []
    for index, atoms in enumerate(frames):
        first_index, neighbor, displacement = neighbor_list("ijD", atoms, max(RADII))
        shell = first_index == mover
        vectors = displacement[shell]
        elements = np.asarray(atoms.get_chemical_symbols())[neighbor[shell]]
        distance = np.linalg.norm(vectors, axis=1)
        nearest = {e: float(distance[elements == e].min()) for e in sorted(set(elements))}
        image_rows.append(dict(slot=key, cohort=cohort, image=index, s=float(coordinates[index]),
                               middle_anchor=index == middle, nearest_by_element_A=nearest,
                               nearest_non_Li_contact_A=min((d for e, d in nearest.items()
                                                            if e != "Li"), default=None),
                               counts_at_3A=dict(Counter(elements[distance <= BASELINE_RADIUS]))))
        if index in (0, middle, len(frames) - 1):
            for radius in RADII:
                sigs[radius, "chemical"].append(signature_from_vectors(vectors, elements, radius))
                sigs[radius, "geometry"].append(signature_from_vectors(vectors,
                                                                       np.full(len(elements), "X"), radius))
    return key, cohort, sigs, image_rows


def compare_anchors(left, right):
    if not left[0] or left[0] != right[0]:
        return None
    errors = [float(np.sqrt(np.mean((a - b)**2))) for a, b in zip(left[1:], right[1:]) if a.size]
    return max(errors)


def compare(first, second, radius, mode):
    a, b = first[2][radius, mode], second[2][radius, mode]
    forward = [compare_anchors(x, y) for x, y in zip(a, b)]
    reverse = [compare_anchors(x, y) for x, y in zip(a, reversed(b))]
    complete = [max(values) for values in (forward, reverse) if None not in values]
    return dict(path_a=first[0], path_b=second[0], reference=second[1], radius_A=radius,
                encoding=mode, middle_gap_A=forward[1],
                all_anchor_gap_A=min(complete, default=None))


def overlaps(comparisons, slots):
    report = []
    for radius, mode, tolerance, cohort in product(RADII, ("chemical", "geometry"), TOLERANCES,
                                                  ("candidate", "H1", "FP-DeErr_relaxed_geometry")):
        current = [r for r in comparisons if (r["radius_A"], r["encoding"], r["reference"])
                   == (radius, mode, cohort)]
        middle = [r for r in current if r["middle_gap_A"] is not None and r["middle_gap_A"] <= tolerance]
        matches = [r for r in current if r["all_anchor_gap_A"] is not None
                   and r["all_anchor_gap_A"] <= tolerance]
        links = [(r["path_a"], r["path_b"]) for r in matches]
        panel_hits = len({s for a, b in links for s in (a, b) if s in slots})
        clusters = len(set(groups(slots, links).values())) if cohort == "candidate" else None
        report.append(dict(radius_A=radius, encoding=mode, tolerance_A=tolerance, reference=cohort,
                           pairs=len(current), middle_anchor_hits=len(middle), all_anchor_hits=len(matches),
                           panel_paths_with_all_anchor_hit=panel_hits, candidate_clusters=clusters))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "geometry", "h1_images", "fp_reference", "sssp_metadata",
                 "local_output", "public_output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--upf-dir", type=Path, action="append", default=[])
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = args.geometry / "geometry_manifest.json"
    freeze = json.loads((args.results / "phase_h2_eval_geometry/summary.json").read_bytes())
    seed_rows = json.loads(checked(manifest, freeze["sha256"][manifest.name]))
    if Counter(r["role"] for r in seed_rows) != {"pilot": 12, "reserve": 8}:
        raise ValueError("Unexpected frozen panel membership")
    candidates = []
    for row in seed_rows:
        seed = args.geometry / f"seed_paths/{row['slot']}.extxyz"
        checked(seed, row["seed_sha256"])
        candidates.append((row["slot"], "candidate", read(seed, ":"), row["moving_Li_index_zero_based"]))
    unmatched = []
    other = list(references(args.results, args.h1_images, args.fp_reference, unmatched))
    elements = {e for r in seed_rows for e in r["chemsys"].split("-")}
    pseudos, archive_status = pseudo_inventory(elements, args.sssp_metadata, args.upf_dir, args.archive)
    qe = [qe_path(row, path[2][0], pseudos) for row, path in zip(seed_rows, candidates)]
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        features = list(workers.map(local_features, candidates + other))
    n = len(candidates)
    panel, reference = features[:n], features[n:]
    pairs = [*combinations(panel, 2), *product(panel, reference)]
    comparisons = [compare(a, b, radius, mode) for a, b in pairs for radius in RADII
                   for mode in ("chemical", "geometry")]
    grid = overlaps(comparisons, [r["slot"] for r in seed_rows])
    flags = defaultdict(list)
    for r in comparisons:
        if r["all_anchor_gap_A"] is not None and r["all_anchor_gap_A"] <= max(TOLERANCES):
            flags[r["path_a"]].append(r)
            if r["reference"] == "candidate":
                flags[r["path_b"]].append(r)
    qualification = [
        {**row, "local_overlap_flags": flags[row["slot"]],
         "protocol_role": ("Protocol Qualification Set" if row["role"] == "pilot" else "reserve"),
         "release_status": "pending_charge_spin_and_numerical_qualification"}
        for row in qe
    ]
    summary = dict(paths=n, qualification_paths=12, reserves=8, production_qualified_paths=0,
                   clean_primary_panel_labeled_paths=0,
                   required_elements=len(elements),
                   upf_status=dict(Counter(row["file_status"] for row in pseudos.values())),
                   archives=archive_status, roles={
                       role: dict(paths=len(group), complete_upf_headers=sum(not r["pending_upf"] for r in group),
                                  odd_neutral_valence=sum(r["neutral_odd_valence"] is True for r in group),
                                  odd_neutral_all_electron=sum(r["neutral_all_electron_count"] % 2 for r in group),
                                  spin_screen_flagged=sum(bool(r["magnetic_element_screen"]) for r in group))
                       for role in ("pilot", "reserve")
                       for group in ([r for r in qe if r["role"] == role],)},
                   one_or_more_pending_upf=sum(bool(r["pending_upf"]) for r in qe),
                   comparison_references=dict(Counter(r[1] for r in reference)),
                   fp_references_with_ambiguous_mover=len(unmatched), local_path_pairs=len(pairs),
                   baseline=[r for r in grid if r["radius_A"] == BASELINE_RADIUS
                             and r["tolerance_A"] == BASELINE_TOLERANCE],
                   source_panel_sha256=digest(manifest), code_sha256=digest(Path(__file__)))
    args.local_output.mkdir(parents=True)
    args.public_output.mkdir(parents=True, exist_ok=True)
    for name, data in (("upf_inventory", pseudos), ("qualification_register", qualification),
                       ("reference_exclusions", unmatched),
                       ("local_image_features", [im for f in features for im in f[3]]),
                       ("local_pair_audit", comparisons)):
        save_json(args.local_output / f"{name}.json", data)
    save_json(args.public_output / "summary.json", summary)
    write(args.public_output / "overlap_sensitivity.csv", grid)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
