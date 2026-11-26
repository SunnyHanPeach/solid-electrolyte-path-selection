"""Audit why the CPU BVSE prescreen cannot score some candidate chemistries."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
from pathlib import Path

from ase.io import read
from ions.data import bvse_data
from ions.decorate import assign_oxidation_states

from audit_project_data import write_csv, write_json
from run_phase_b4_source_pilot import archive_geometry, read_csv


def check_material(entry):
    material, rows, record = entry
    result = {"material_id": material, "chemsys": rows[0]["chemsys"], "failed_paths": len(rows),
              "original_errors": " | ".join(sorted({row["error"] for row in rows})),
              "frozen_status": "", "extended_search_status": "", "unsupported_anions": ""}
    if record is None:
        result["frozen_status"] = "missing_parent"
        return result
    _, blob = record
    atoms = read(io.StringIO(blob.decode("utf-8")), format="extxyz")
    parameters = bvse_data["Li"][1]
    for name, limit in (("frozen_status", 10000), ("extended_search_status", 100000)):
        try:
            decorated = assign_oxidation_states(atoms.copy(), max_permutations=limit)
        except (ValueError, KeyError, RuntimeError) as exc:
            result[name] = f"oxidation_failed: {type(exc).__name__}: {exc}"
            continue
        ox = decorated.arrays["oxi_states"]
        if abs(float(ox.sum())) > 1e-5:
            result[name] = "nonneutral_assignment"
            continue
        unsupported = sorted({f"{element}:{charge:g}" for element, charge in zip(decorated.symbols, ox)
                              if charge < 0 and charge not in parameters.get(element, {})})
        result[name] = "unsupported_negative_species" if unsupported else "parameterized"
        result["unsupported_anions"] = ";".join(unsupported)
        if name == "frozen_status":
            result["extended_search_status"] = "already_assigned"
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "parents", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in read_csv(args.screen):
        if row["status"] == "failed":
            groups[row["material_id"]].append(row)
    _, parents, _ = archive_geometry(args.parents, "BVEL13k", "material_id", set(groups), set(groups))
    tasks = [(material, rows, parents.get(material)) for material, rows in sorted(groups.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check_material, tasks, chunksize=2))
    args.output.mkdir()
    write_csv(args.output / "oxidation_failure_audit.csv", results)
    summary = dict(unique_materials=len(groups), path_failures=sum(map(len, groups.values())),
                   frozen_material_status=Counter(row["frozen_status"] for row in results),
                   extended_material_status=Counter(row["extended_search_status"] for row in results),
                   unsupported_negative_species=Counter(species for row in results
                                                        for species in row["unsupported_anions"].split(";") if species),
                   recorded_primary_results_unchanged=True)
    write_json(args.output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
