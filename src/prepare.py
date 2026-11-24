"""Freeze the E/F diagnostic from the previously hashed eight-path panel."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


CHECKPOINT = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"
PANEL = (("mp-3368_0_0_1_0_0", "mp-777350_7_4_1_0_-1"),
         ("mp-7603_0_2_0_0_0", "mp-776177_0_4_1_0_0"))
SETTINGS = dict(seeds=[17, 29, 43], epochs=5, batch_size=2, lr=0.0001,
                energy_weight=1.0, forces_weight=100.0, clip_norm=10.0)
# These directions come from the completed energy-only pilot and precede E/F fitting.
ENERGY_ONLY_SIGNS = dict(zip((edge for pair in PANEL for edge in pair), [1, 1, 1, -1]))


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prepare(source, destination):
    frozen_path = source / "frozen_design.json"
    frozen = json.loads(frozen_path.read_bytes())
    for name, sha in frozen["file_sha256"].items():
        if digest(source / name) != sha:
            raise ValueError(f"Changed source input: {name}")
    source_rows = rows(source / "frames.csv")
    specs = {r["edge_id"]: r for r in rows(source / "selection.csv")}
    allowed = {edge for pair in PANEL for edge in pair}
    destination.mkdir()
    kept, mapped = [], {}
    with (source / "geometries.extxyz").open("rb") as xyz, (
            destination / "geometries.extxyz").open("wb") as out:
        for i, row in enumerate(source_rows):
            size = xyz.readline()
            title = xyz.readline()
            if int(size) != int(row["n_atoms"]) or not title:
                raise ValueError(f"Malformed frame {i}")
            body = [xyz.readline() for _ in range(int(size))]
            if row["role"] in {"initial", "validation"} or row["edge_id"] in allowed:
                mapped[i] = len(kept)
                kept.append(row)
                out.writelines([size, title, *body])
        if xyz.read(1):
            raise ValueError("Unindexed source geometry")
    baseline = [mapped[i] for i in frozen["baseline_rows"]]
    cases = {}
    for edge in sorted(allowed):
        original = frozen["candidate_rows"][edge]
        replay = frozen["replay"][edge][0]
        candidate_idx, replay_idx = ([mapped[i] for i in group] for group in (original, replay))
        n_atoms = sum(int(kept[i]["n_atoms"]) for i in candidate_idx)
        if (len(candidate_idx) != len(replay_idx) or not set(replay_idx).issubset(baseline)
                or sum(int(kept[i]["n_atoms"]) for i in replay_idx) != n_atoms):
            raise ValueError(f"Invalid replay exposure for {edge}")
        cases[edge] = dict(candidate=candidate_idx, replay=replay_idx, replay_draw_number=1,
                           replay_pool=specs[edge]["replay_pool"], added_atom_images=n_atoms,
                           added_frames=len(original))
    refs = rows(source / "validation_paths.csv")
    fit_materials = {r["material_id"] for r in kept if r["role"] != "validation"}
    val_materials = {r["material_id"] for r in refs}
    if len(baseline) != 278 or len(refs) != 241 or len(val_materials) != 124 or fit_materials & val_materials:
        raise ValueError("Unexpected baseline or validation partition")
    write_csv(destination / "frames.csv", kept)
    write_csv(destination / "validation_paths.csv", refs)
    files = {p.name: digest(p) for p in sorted(destination.iterdir())}
    design = dict(source_design_sha256=digest(frozen_path), checkpoint_sha256=CHECKPOINT,
                  input_sha256=files, settings=SETTINGS, pairs=PANEL, cases=cases, d0=baseline,
                  energy_only_case_directions=ENERGY_ONLY_SIGNS, energy_only_pair_directions=[1, -1])
    (destination / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(cases)} candidates, {len(baseline)} baseline images and {len(refs)} validation paths")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
