"""Prepare and archive a CPU-only fixed-geometry QE spin pilot for mp-1191903."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5, sha256
import io
import json
from math import ceil
from pathlib import Path
import re
import sys

import ase
import numpy as np
from ase.geometry import find_mic
from ase.io import read, write


EDGE = "mp-1191903_0_2_0_0_0"
SPECIES = ("In", "Li", "S", "Si")
SSSP_METADATA_MD5 = "3153c4b20fc90a44fba0236627525644"
SMEARING_EV = 0.10
RY_EV = 13.605693123
TOL = 1e-8


def numeric(x):
    return float(x.replace("D", "E").replace("d", "e"))


def attributes(s):
    return dict(re.findall(r"([\w.-]+)\s*=\s*[\"'](.*?)[\"']", s, re.S))


def read_upf(element, reference, upfs, stage):
    raw = (upfs / reference["filename"]).read_bytes()
    digest = md5(raw).hexdigest()
    if digest != reference["md5"]:
        raise ValueError(f"{element} UPF hash differs from the SSSP release")

    body = raw.decode("utf-8", errors="replace")
    tag = re.search(r"<PP_HEADER\b([^>]*)>", body, re.S)
    if tag is None:
        raise ValueError(f"No PP_HEADER in {element} UPF")
    header = attributes(tag[1])
    if header:
        if header.get("element", "").strip() != element:
            raise ValueError(f"Wrong PP_HEADER element for {element}")
        functional = header["functional"].strip()
        valence = numeric(header["z_valence"])
        relativity = header.get("relativistic", "").strip()
        spin_orbit = header.get("has_so", "").casefold() in ("t", "true")
        pseudo_type = header["pseudo_type"]
        recommended = [numeric(header.get(k, "0")) for k in ("wfc_cutoff", "rho_cutoff")]
        orbitals = [attributes(s) for s in re.findall(r"<PP_CHI\.\d+\b([^>]*)>", body, re.S)]
        orbitals = {r["label"]: numeric(r["occupation"]) for r in orbitals if "occupation" in r}
        version = re.search(r"<UPF\s+version=[\"']([^\"']+)", body)[1]
    else:
        legacy = body[tag.end():body.index("</PP_HEADER>", tag.end())]
        lines = [line.strip() for line in legacy.splitlines() if line.strip()]
        if not any(re.fullmatch(rf"{element}\s+Element", x) for x in lines):
            raise ValueError(f"Wrong legacy header element for {element}")
        functional = next(x.split("Exchange-Correlation", 1)[0].strip()
                          for x in lines if "Exchange-Correlation" in x)
        valence = numeric(next(x.split()[0] for x in lines if "Z valence" in x))
        relativity = ("nonrelativistic" if "Non-Relativistic Calculation" in body
                      else "scalar" if "Scalar-Relativistic Calculation" in body else "unresolved")
        spin_orbit = False if relativity != "unresolved" else None
        pseudo_type = "USPP" if any("Ultrasoft pseudopotential" in x for x in lines) else "unresolved"
        recommended = [numeric(x) for x in next(x for x in lines if "Suggested cutoff" in x).split()[:2]]
        orbitals = {r[0]: numeric(r[-1]) for x in lines if
                    (r := x.split()) and len(r) == 3 and re.fullmatch(r"\d+[SPDF]", r[0])}
        version = "UPF v1"

    codes = functional.upper().split()
    if not ("PBE" in codes or {"PBX", "PBC"} <= set(codes)):
        raise ValueError(f"Unexpected exchange correlation for {element}: {functional}")
    if not orbitals or spin_orbit is not False or pseudo_type not in ("USPP", "NC", "PAW"):
        raise ValueError(f"{element} UPF is missing orbital or relativity information")
    expected_semicore = {"In": "4D", "Li": "1S"}.get(element)
    if expected_semicore and orbitals.get(expected_semicore, 0) <= 0:
        raise ValueError(f"Expected occupied {element} semicore channel was absent")

    (stage / "pseudo" / reference["filename"]).write_bytes(raw)
    return dict(filename=reference["filename"], sssp_md5=digest, sha256=sha256(raw).hexdigest(),
                source_release=reference["pseudopotential"], format=version,
                xc="PBE", reported_xc=functional, relativity=relativity, spin_orbit=spin_orbit,
                pseudo_type=pseudo_type, z_valence=valence, reference_orbital_occupations=orbitals,
                explicit_semicore=expected_semicore, header_cutoffs_Ry=recommended,
                sssp_cutoffs_Ry=[reference["cutoff_wfc"], reference["cutoff_rho"]])


def prepare_input(atoms, role, image, spin, pseudopotentials, cutoffs, output):
    data = atoms.copy()
    data.calc = None
    data.set_constraint()
    data.set_initial_magnetic_moments(np.zeros(len(data)))
    identifier = f"{role}_{spin}"
    system = dict(input_dft="PBE", ecutwfc=cutoffs[0], ecutrho=cutoffs[1],
                  nspin=1 if spin == "nspin1" else 2, tot_charge=0,
                  occupations="smearing", smearing="gaussian", degauss=SMEARING_EV / RY_EV, nosym=True)
    if spin == "spin1muB":
        system["tot_magnetization"] = 1.0
    options = {
        "control": dict(calculation="scf", prefix=f"b4_mp1191903_{identifier}", tprnfor=True,
                        tstress=True, pseudo_dir="./pseudo", outdir="./scratch"),
        "system": system,
        "electrons": dict(conv_thr=1e-8, electron_maxstep=80, mixing_beta=.3),
    }
    stream = io.StringIO()
    write(stream, data, format="espresso-in", pseudopotentials=pseudopotentials, input_data=options, kpts=None)
    text = stream.getvalue()
    text = re.sub(r"(?m)^[ \t]*starting_magnetization\(\d+\)[ \t]*=.*\n", "", text)
    text = re.sub(r"(?m)^&(?:IONS|CELL|FCP|RISM)[ \t]*\n[ \t]*/[ \t]*\n", "", text)
    parsed = read(io.StringIO(text), format="espresso-in")
    if not (np.array_equal(parsed.numbers, atoms.numbers)
            and np.allclose(parsed.cell, atoms.cell, atol=1e-9, rtol=0)
            and np.allclose(parsed.positions, atoms.positions, atol=1e-9, rtol=0)):
        raise ValueError(f"{identifier} round trip fails")
    if ("tot_magnetization" in text) != (spin == "spin1muB") or "starting_magnetization" in text:
        raise ValueError(f"{identifier} spin contract changed")
    spin_assignment = re.search(r"(?m)^\s*nspin\s*=\s*(\d+)", text)
    if spin_assignment is None or int(spin_assignment[1]) != system["nspin"] or "K_POINTS gamma" not in text:
        raise ValueError(f"{identifier} spin or k-point format changed")
    name = identifier + ".in"
    raw = text.encode("utf-8")
    if b"\r" in raw:
        raise ValueError("Input has a non-Linux line ending")
    (output / name).write_bytes(raw)
    return dict(role=role, original_image=image, spin=spin, nspin=system["nspin"],
                tot_magnetization_muB=system.get("tot_magnetization"), input_file=name,
                sha256=sha256(raw).hexdigest())


def main():
    p = argparse.ArgumentParser()
    for key in ("trajectory", "pilot", "pre02_manifest", "sssp_metadata", "upfs", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")

    raw = args.trajectory.read_bytes()
    trajectory_hash = sha256(raw).hexdigest()
    pre02_bytes = args.pre02_manifest.read_bytes()
    pre02 = json.loads(pre02_bytes)
    if trajectory_hash != pre02["candidate_paths"][EDGE]["geometry_sha256"]:
        raise ValueError("Chosen geometry does not match the Pre-02 audited file")
    pilot_bytes = args.pilot.read_bytes()
    records = json.loads(pilot_bytes)
    record = next(r for r in records["results"] if r["edge_id"] == EDGE)
    if not record["converged"]:
        raise ValueError("The BVSE source pilot did not converge")
    images = read(io.StringIO(raw.decode()), ":", format="extxyz")
    first = images[0]
    moving = record["migrating_Li_0based"]
    counts = dict(Counter(first.get_chemical_symbols()))
    expected = {"In": 16, "Li": 15, "S": 48, "Si": 8}
    if counts != expected or len(images) != 5 or any(
        not np.array_equal(a.numbers, first.numbers) or not np.allclose(a.cell.array, first.cell.array, atol=TOL, rtol=0)
        or not a.pbc.all() or not np.isfinite(a.positions).all() for a in images
    ) or first.numbers[moving] != 3:
        raise ValueError("Audited atom ordering, count, periodic cell or Li site changed")
    peak = max(range(1, len(images) - 1), key=lambda i: record["final"][i]["bvse_energy_eV"])
    if record["final"][peak]["bvse_energy_eV"] <= max(record["final"][j]["bvse_energy_eV"] for j in (0, len(images) - 1)):
        raise ValueError("No positive BVSE internal peak in the selected path")
    fixed = np.arange(len(first)) != moving
    shifted = find_mic(images[peak].positions[fixed] - first.positions[fixed], first.cell, first.pbc)[1].max()
    if shifted > TOL:
        raise ValueError("The scaffold of the fixed-geometry pilot differs at the BVSE peak")

    metadata = args.sssp_metadata.read_bytes()
    if md5(metadata).hexdigest() != SSSP_METADATA_MD5:
        raise ValueError("Unexpected official SSSP catalogue")
    reference = json.loads(metadata)
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "pseudo").mkdir()
    with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        info = dict(zip(SPECIES, pool.map(
            lambda v: read_upf(v, reference[v], args.upfs, args.output), SPECIES
        )))
        cutoffs = [ceil(max(max(p[s][i] for s in ("sssp_cutoffs_Ry", "header_cutoffs_Ry"))
                            for p in info.values())) for i in range(2)]
        valence = sum(counts[element] * info[element]["z_valence"] for element in counts)
        if not valence.is_integer() or int(valence) % 2 != 1:
            raise ValueError("Expected a neutral vacancy geometry with an odd valence-electron count")
        slots = [(role, j, spin) for role, j in (("endpoint", 0), ("peak", peak)) for spin in ("nspin1","spin1muB")]
        filenames = {e: info[e]["filename"] for e in SPECIES}
        jobs = list(pool.map(lambda x: prepare_input(images[x[1]], *x, filenames, cutoffs, args.output), slots))

    manifest = dict(edge_id=EDGE, geometry_source_sha256=trajectory_hash,
                    pre02_manifest_sha256=sha256(pre02_bytes).hexdigest(),
                    bvse_pilot_sha256=sha256(pilot_bytes).hexdigest(),
                    environment=dict(python=sys.version.split()[0], ase=ase.__version__, numpy=np.__version__),
                    geometry_type="fixed_BVSE_preconditioned_5_image_path",
                    atom_counts=counts, migrating_Li_0based=moving, total_charge_e=0,
                    qe_valence_electrons=int(valence), fixed_spin_population=[(int(valence)+1)//2, (int(valence)-1)//2],
                    cell_volume_A3=float(first.cell.volume), first_atom_count=len(first),
                    sssp_release="SSSP 1.3.0 PBE Efficiency", sssp_metadata_md5=SSSP_METADATA_MD5,
                    b4_bridge_qe="7.3.1", k_points="Gamma", smearing="Gaussian", smearing_eV=SMEARING_EV,
                    chosen_cutoffs_Ry=cutoffs, electronic_conv_thr_Ry=1e-8,
                    selected_images={"endpoint": 0,"bvse_peak": peak}, pseudopotentials=info,
                    jobs=jobs, started_dft=False, reserved_test_labels_read=0,
                    gate=dict(energy_screen_meV=10, force_component_rmse_screen_eV_A=.02,
                              migrating_li_force_difference_norm_screen_eV_A=.02,
                              two_endpoint_jobs_required_before_peak=True))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Prepared {len(jobs)} jobs, {int(valence)} valence electrons, cutoffs {cutoffs}, BVSE peak image {peak}")


if __name__ == "__main__":
    main()
