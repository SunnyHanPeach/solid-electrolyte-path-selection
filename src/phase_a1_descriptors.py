"""Build a CPU pilot database of separate Phase A1 Li-centered representations."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
from ase import Atoms

from phase_a0_geometry import PATHS, STAGES, csv_rows, exact_sha, export_csv, minimum_vectors
from phase_a01_coverage import load_inputs

METHODS = {
    "rdf_geometry": ("geometry", "cosine"),
    "soap_geometry": ("geometry", "cosine"),
    "coordination_hard": ("chemistry", "euclidean"),
    "coordination_smooth": ("chemistry", "euclidean"),
    "rdf_elements": ("chemistry", "cosine"),
    "soap_elements": ("chemistry", "cosine"),
    "mace_invariants": ("model", "cosine"),
}
GRID = np.arange(1., 5., .15)
SOAP_SETTINGS = dict(r_cut=4., n_max=5, l_max=3, sigma=.4, periodic=True, dtype="float64")
UNIT_TOL = 1e-5
D0_WEIGHT = "D0_17.pt"
D0_WEIGHT_SHA256 = "c0ed6bed7c168f24b0c9392dd2d786c10f209072493858aa04dbcc1ab3c4e69a"


def taper(r, start, stop):
    u = np.clip((r-start)/(stop-start),0.,1.)
    return .5*(1.+np.cos(np.pi*u))


def radial_descriptors(atoms, centers, species):
    """Return Li-centered counts and Gaussian shell densities for each selected center."""
    vec = minimum_vectors(atoms.positions[None]-atoms.positions[centers][:,None], atoms.cell, atoms.pbc)
    r = np.linalg.norm(vec,axis=-1)
    numbers = np.asarray(atoms.get_chemical_symbols())
    bins = (numbers[:,None] == np.asarray(species)[None,:]).astype(float)
    mask = r > 1e-10
    hard = ((r<=3.) & mask) @ bins
    smooth = (taper(r, 2.6, 3.4)*mask) @ bins
    sigma = .12
    gaussian = np.exp(-.5*((r[:,:,None]-GRID)/sigma)**2)
    gaussian *= (taper(r,4.6,5.)*mask)[:,:,None]
    gaussian /= (np.sqrt(2.*np.pi)*sigma)*(4.*np.pi*GRID**2)
    per_element = np.einsum("cag,ae->ceg",gaussian,bins,optimize=True)
    return dict(coordination_hard=hard, coordination_smooth=smooth,
                rdf_geometry=gaussian.sum(axis=1), rdf_elements=per_element.reshape(len(centers),-1))


def make_extractors(args, species):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD","1")
    from dscribe.descriptors import SOAP
    from mace.calculators import MACECalculator
    import torch

    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    with ZipFile(args.checkpoints) as archive:
        models = json.loads(archive.read("models.json"))
        blob = archive.read(D0_WEIGHT)
    weight_sha = exact_sha(blob)
    expected = models["weights"][D0_WEIGHT]["sha256"]
    if weight_sha != expected or weight_sha != D0_WEIGHT_SHA256:
        raise ValueError("D0 checkpoint bytes differ from the audited weights")
    model = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    calc = MACECalculator(models=model, device="cpu", default_dtype="float64")
    for model in calc.models:
        model.eval()
    geometry = SOAP(species=["H"], **SOAP_SETTINGS)
    chemistry = SOAP(species=species, compression={"mode":"mu1nu1","species_weighting":None}, **SOAP_SETTINGS)
    layers = int(calc.models[0].num_interactions)

    def extract(atoms, centers):
        values = radial_descriptors(atoms, centers, species)
        clean = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        anonymous = clean.copy()
        anonymous.numbers[:] = 1
        values["soap_geometry"] = np.atleast_2d(geometry.create(anonymous, centers=centers.tolist()))
        values["soap_elements"] = np.atleast_2d(chemistry.create(clean, centers=centers.tolist()))
        values["mace_invariants"] = calc.get_descriptors(atoms=clean, invariants_only=True,
                                                            num_layers=layers)[centers]
        if any(v.ndim!=2 or len(v)!=len(centers) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Invalid Li feature dimensions or values")
        return values

    return extract, dict(checkpoint=D0_WEIGHT, checkpoint_sha256=weight_sha, layers=layers,
                         mace="scalar invariants from each interaction layer",
                         ase=version("ase"), torch=torch.__version__, dscribe=version("dscribe"),
                         runtime_dtype="float64")


def invariance(queries, extract):
    rows = []
    rng = np.random.default_rng(20261125)
    for atoms,mover,key in queries:
        if key["image"] != (4 if key["edge_id"] == PATHS[2] else 3):
            continue
        center = np.array([mover])
        base = extract(atoms,center)
        rotated = atoms.copy()
        q,_ = np.linalg.qr(rng.normal(size=(3,3)))
        q[:,-1] *= np.linalg.det(q)
        rotated.set_cell(atoms.cell.array@q)
        rotated.positions[:] = atoms.positions@q
        perm = rng.permutation(len(atoms))
        changed_mover = np.flatnonzero(perm==mover)
        shifted = atoms.copy()
        shifted.positions[:] += [.7,1.2,-.4]
        shifted.wrap()
        for check, transformed, new_center in (
            ("repeat",atoms,center),("rigid_rotation",rotated,center),
            ("permutation",atoms[perm],changed_mover),("translation_and_wrap",shifted,center),
        ):
            altered = extract(transformed,new_center)
            for method, values in base.items():
                d = float(np.linalg.norm(values-altered[method])/max(np.linalg.norm(values),1.))
                rows.append(dict(key, method=method, check=check, relative_L2=d, tolerance=UNIT_TOL,
                                 passes=int(d<=UNIT_TOL)))
    return rows


def load_representatives(path, frames):
    table = csv_rows(path.read_bytes())
    result = {(int(i),int(li)) for r in table if float(r["radius_A"])==3.
              for i,li in json.loads(r["dedup_representatives"])}
    available = {(frame,int(li+1)) for frame, atoms in enumerate(frames,1)
                 for li in np.flatnonzero(atoms.numbers==3)}
    if not result <= available or len(result)!=sum(int(r["dedup_Li"]) for r in table if float(r["radius_A"])==3.):
        raise ValueError("A0.1 representative IDs do not match D0")
    return result


def vector_distances(query, bank, metric):
    if metric=="euclidean":
        return np.linalg.norm(bank-query,axis=1)
    denom = np.linalg.norm(query)*np.linalg.norm(bank,axis=1)
    cosine = np.divide(bank@query,denom,out=np.zeros(len(bank)),where=denom>1e-20)
    cosine[(denom<=1e-20) & (np.linalg.norm(query)<=1e-10) & (np.linalg.norm(bank,axis=1)<=1e-10)] = 1.
    return 1.-np.clip(cosine,-1.,1.)


def reference_profiles(features, refs, keys, reference_ids, use):
    materials = np.asarray([v[2] for v in reference_ids])
    material_indices = [np.flatnonzero(materials==mat) for mat in sorted(set(materials))]
    rows = []
    for method, (family, metric) in METHODS.items():
        for i, query in enumerate(features[method]):
            d = vector_distances(query, refs[method], metric)
            nearest_raw = int(np.argmin(d))
            selected = np.flatnonzero(use)
            nearest_reduced = int(selected[np.argmin(d[selected])])
            frame, li, material = reference_ids[nearest_reduced]
            rows.append(dict(
                keys[i], feature_row_0based=i, representation=method, family=family, distance=metric,
                raw_nearest=float(d[nearest_raw]), dedup_nearest=float(d[nearest_reduced]),
                median_material_nearest=float(np.median([d[j].min() for j in material_indices])),
                dedup_nearest_D0_frame=frame, dedup_nearest_D0_Li=li, dedup_nearest_material=material,
            ))
    return rows


def plot_profiles(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(METHODS), len(PATHS), figsize=(11, 15), sharex=True,
                            sharey="row", layout="constrained")
    for i, (method, (_, metric)) in enumerate(METHODS.items()):
        for j, case in enumerate(PATHS):
            ax = axs[i,j]
            for stage, style in zip(STAGES,("-o","--s")):
                records = [r for r in rows if (r["representation"],r["edge_id"],r["stage"])==(method,case,stage)]
                ax.plot([r["reaction_coordinate"] for r in records],
                        [r["raw_nearest"] for r in records], style, markersize=3, label=stage)
            if not j:
                ax.set_ylabel(f"{method.replace('_',' ')}\n{metric}")
            if not i:
                ax.set_title(case.split("_")[0])
            ax.grid(alpha=.2)
    axs[0,-1].legend(fontsize=8)
    fig.supxlabel("Normalized reaction coordinate")
    fig.suptitle("Nearest D₀ distance by representation")
    fig.savefig(output/"novelty_profiles.png",dpi=143)
    plt.close(fig)


def run(args):
    start = time.perf_counter()
    frames, metadata, queries = load_inputs(args.neb,args.index,args.bundle,args.a0)
    species = sorted({symbol for a in [*frames, *(a for a,_,_ in queries)] for symbol in a.get_chemical_symbols()})
    reduced = load_representatives(args.a01/"reference_material_distribution.csv",frames)
    extract, provenance = make_extractors(args,species)
    invariance_rows = invariance(queries,extract)
    if any(not r["passes"] for r in invariance_rows):
        failed = [row for row in invariance_rows if not row["passes"]]
        raise ValueError(f"Descriptor invariance failed: {failed[:4]}")
    print(f"{len(invariance_rows)} representation checks passed",flush=True)
    features, refs = defaultdict(list), defaultdict(list)
    keys = [k for _,_,k in queries]
    for atoms, mover, _ in queries:
        for method, v in extract(atoms,np.array([mover])).items():
            features[method].append(v)
    reference_ids = []
    for frame,(atoms,row) in enumerate(zip(frames,metadata),1):
        centers = np.flatnonzero(atoms.numbers==3)
        for method,v in extract(atoms,centers).items():
            refs[method].append(v)
        reference_ids.extend((frame,int(li+1),row["material_id"]) for li in centers)
        if frame%50==0:
            print(f"D0 geometries {frame}/{len(frames)}",flush=True)
    features = {k:np.concatenate(v) for k,v in features.items()}
    refs = {k:np.concatenate(v) for k,v in refs.items()}
    use = np.asarray([(i,li) in reduced for i,li,_ in reference_ids])
    if len(reference_ids)!=9517 or int(use.sum())!=2469:
        raise ValueError("Unexpected reference counts")
    profiles = reference_profiles(features,refs,keys,reference_ids,use)
    args.output.mkdir(parents=True,exist_ok=False)
    export_csv(args.output/"feature_index.csv", [dict(k,feature_row_0based=i,
               DFT_preselectable=int(k["stage"]=="init")) for i,k in enumerate(keys)])
    export_csv(args.output/"invariance_tests.csv",invariance_rows)
    export_csv(args.output/"novelty_profiles.csv",profiles)
    np.savez_compressed(args.output/"path_features.npz",**features)
    plot_profiles(args.output,profiles)
    provenance.update(dict(methods=METHODS, shapes={k:list(v.shape) for k,v in features.items()},
                           species=species, rdf_grid_A=GRID.tolist(), soap_settings=SOAP_SETTINGS,
                           coordination_A={"hard":3.,"smooth":[2.6,3.4]}, rdf_sigma_A=.12,
                           rdf_upper_taper_A=[4.6,5.], chemical_soap_compression="mu1nu1",
                           reference_dedup_radius_A=3., references_raw=len(reference_ids),
                           references_dedup=int(use.sum()), materials=len(set(x[2] for x in reference_ids)),
                           source_stage_policy={"init":"pre-DFT","relaxed":"diagnostic only"},
                           A0_manifest_sha256=exact_sha((args.a0/"geometry_manifest.csv").read_bytes()),
                           A01_materials_sha256=exact_sha((args.a01/"reference_material_distribution.csv").read_bytes()),
                           elapsed_seconds=round(time.perf_counter()-start,3)))
    (args.output/"manifest.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(f"Finished {len(keys)} path images and {len(reference_ids)} D0 environments",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("neb","index","bundle","a0","a01","checkpoints","output"):
        cli.add_argument(name,type=Path)
    cli.add_argument("--threads",type=int,default=4)
    run(cli.parse_args())
