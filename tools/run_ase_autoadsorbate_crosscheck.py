from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import (
    bcc100,
    bcc110,
    bcc111,
    bulk,
    diamond100,
    diamond111,
    fcc100,
    fcc110,
    fcc111,
    fcc211,
    hcp0001,
    hcp10m10,
    surface,
)
from ase.cluster.icosahedron import Icosahedron
from ase.collections import g2
from ase.io import write
from ase.spacegroup import crystal

from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig, build_site_dictionary
from adsorption_ensemble.surface import SurfacePreprocessor
from adsorption_ensemble.visualization import plot_inequivalent_sites_2d, plot_site_centers_only, plot_surface_primitives_2d
from tests.chemistry_cases import get_test_adsorbate_cases


@dataclass
class PlacementStats:
    attempted: int = 0
    succeeded: int = 0
    zero_primitive: int = 0
    zero_pose: int = 0
    soft_clash: int = 0
    exception: int = 0
    min_interatomic_distance: float = 1.0e9

    def update_distance(self, d: float) -> None:
        self.min_interatomic_distance = float(min(self.min_interatomic_distance, d))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.min_interatomic_distance > 1.0e8:
            out["min_interatomic_distance"] = None
        return out


def _with_surface_vacancy(base: Atoms) -> Atoms:
    slab = base.copy()
    z = np.asarray(slab.get_positions(), dtype=float)[:, 2]
    del slab[int(np.argmax(z))]
    slab.info.pop("adsorbate_info", None)
    return slab


def _with_surface_adatom(base: Atoms, symbol: str = "Pt") -> Atoms:
    slab = base.copy()
    pos = np.asarray(slab.get_positions(), dtype=float)
    idx = int(np.argmax(pos[:, 2]))
    add = Atoms(symbols=[symbol], positions=[pos[idx] + np.asarray([0.0, 0.0, 1.9], dtype=float)])
    out = slab + add
    out.set_cell(slab.cell)
    out.set_pbc(slab.get_pbc())
    out.info.pop("adsorbate_info", None)
    return out


def _with_cluster_interface(base: Atoms, symbol: str = "Pt") -> Atoms:
    slab = base.copy()
    cluster = Icosahedron(symbol, 1).copy()
    cpos = np.asarray(cluster.get_positions(), dtype=float)
    cpos -= np.mean(cpos, axis=0, keepdims=True)
    spos = np.asarray(slab.get_positions(), dtype=float)
    center_xy = np.mean(spos[:, :2], axis=0)
    cpos[:, 0] += center_xy[0]
    cpos[:, 1] += center_xy[1]
    cpos[:, 2] += float(np.max(spos[:, 2])) + 2.2
    cluster.set_positions(cpos)
    out = slab + cluster
    out.set_cell(slab.cell)
    out.set_pbc(slab.get_pbc())
    out.info.pop("adsorbate_info", None)
    return out


def _replace_top_layer_atoms(base: Atoms, from_symbol: str, to_symbol: str, frac: float) -> Atoms:
    slab = base.copy()
    z = np.asarray(slab.get_positions(), dtype=float)[:, 2]
    z_top = float(np.max(z))
    top = [i for i, zi in enumerate(z) if (z_top - zi) < 1.0]
    n_swap = max(1, int(round(len(top) * float(frac))))
    for i in top[:n_swap]:
        if slab[i].symbol == from_symbol:
            slab[i].symbol = to_symbol
    slab.info.pop("adsorbate_info", None)
    return slab


def build_slab_suite() -> dict[str, Atoms]:
    slabs: dict[str, Atoms] = {}
    slabs["Pt_fcc111"] = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
    slabs["Pt_fcc100"] = fcc100("Pt", size=(4, 4, 4), vacuum=12.0)
    slabs["Pt_fcc110"] = fcc110("Pt", size=(4, 4, 4), vacuum=12.0)
    slabs["Pt_fcc211"] = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
    slabs["Fe_bcc100"] = bcc100("Fe", size=(4, 4, 4), vacuum=12.0)
    slabs["Fe_bcc110"] = bcc110("Fe", size=(4, 4, 4), vacuum=12.0)
    slabs["Fe_bcc111"] = bcc111("Fe", size=(4, 4, 4), vacuum=12.0)
    slabs["Ru_hcp0001"] = hcp0001("Ru", size=(4, 4, 4), vacuum=12.0)
    slabs["Ru_hcp10m10"] = hcp10m10("Ru", size=(4, 4, 4), vacuum=12.0)
    slabs["C_diamond100"] = diamond100("C", size=(4, 4, 6), vacuum=12.0)
    slabs["C_diamond111"] = diamond111("C", size=(4, 4, 6), vacuum=12.0)

    rutile = crystal(
        symbols=["Ti", "O"],
        basis=[(0.0, 0.0, 0.0), (0.305, 0.305, 0.0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )
    slabs["TiO2_110"] = surface(rutile, (1, 1, 0), layers=6, vacuum=12.0).repeat((2, 2, 1))
    slabs["MgO_100"] = surface(bulk("MgO", "rocksalt", a=4.213), (1, 0, 0), layers=6, vacuum=12.0).repeat((2, 2, 1))

    slabs["CuNi_fcc111_alloy"] = _replace_top_layer_atoms(fcc111("Cu", size=(4, 4, 4), vacuum=12.0), "Cu", "Ni", frac=0.25)
    slabs["Pt_fcc111_vacancy"] = _with_surface_vacancy(fcc111("Pt", size=(4, 4, 4), vacuum=12.0))
    slabs["Pt_fcc111_adatom"] = _with_surface_adatom(fcc111("Pt", size=(4, 4, 4), vacuum=12.0), symbol="Pt")
    slabs["Pt_fcc111_cluster_interface"] = _with_cluster_interface(fcc111("Pt", size=(4, 4, 4), vacuum=12.0), symbol="Pt")
    return slabs


def build_molecule_suite(max_atoms: int = 40) -> dict[str, Atoms]:
    out: dict[str, Atoms] = {}
    for name in sorted(g2.names):
        try:
            a = g2[name].copy()
        except Exception:
            continue
        if len(a) <= int(max_atoms):
            out[f"g2_{name}"] = a
    out.update(get_test_adsorbate_cases())
    return out


def _run_our_site_pipeline(slab: Atoms, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
    raw = PrimitiveBuilder().build(slab, ctx)
    z = slab.get_atomic_numbers().astype(float)
    atom_features = (z / (np.max(z) + 1.0e-12)).reshape(-1, 1)
    emb = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.20)).fit_transform(
        slab=slab,
        primitives=list(raw),
        atom_features=atom_features,
    )

    raw_counts = Counter([str(p.kind) for p in emb.primitives])
    basis_counts = Counter([str(p.kind) for p in emb.basis_primitives])
    _write_json(out_dir / "raw_site_dictionary.json", build_site_dictionary(emb.primitives))
    _write_json(out_dir / "selected_site_dictionary.json", build_site_dictionary(emb.basis_primitives))
    plot_surface_primitives_2d(slab=slab, context=ctx, primitives=emb.primitives, filename=out_dir / "sites.png")
    plot_site_centers_only(slab=slab, primitives=emb.primitives, filename=out_dir / "sites_only.png")
    plot_inequivalent_sites_2d(slab=slab, primitives=emb.primitives, filename=out_dir / "sites_inequivalent.png")

    return {
        "n_atoms": int(len(slab)),
        "n_surface_atoms": int(len(ctx.detection.surface_atom_ids)),
        "n_raw_primitives": int(len(emb.primitives)),
        "n_basis_primitives": int(len(emb.basis_primitives)),
        "raw_counts": dict(raw_counts),
        "basis_counts": dict(basis_counts),
        "surface_atom_ids": [int(i) for i in ctx.detection.surface_atom_ids],
        "primitives": emb.basis_primitives,
    }


def _ase_reference_counts(slab: Atoms) -> dict[str, Any]:
    info = slab.info.get("adsorbate_info", {})
    sites = info.get("sites", {}) if isinstance(info, dict) else {}
    if not isinstance(sites, dict) or not sites:
        return {"status": "no_reference_sites"}
    expected = {"ontop": 0, "bridge": 0, "hollow": 0}
    for name in sites.keys():
        lname = str(name).lower()
        if lname == "ontop":
            expected["ontop"] += 1
        elif "bridge" in lname:
            expected["bridge"] += 1
        elif lname in {"fcc", "hcp", "hollow"}:
            expected["hollow"] += 1
    return {"status": "ok", "expected_mapped_counts": expected}


def _autoadsorbate_site_counts(slab: Atoms) -> dict[str, Any]:
    try:
        import autoadsorbate as aa  # type: ignore
    except Exception as exc:
        return {"status": "import_error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        surf = aa.Surface(slab.copy(), mode="slab", precision=0.35, touch_sphere_size=3.0)
        total_before = int(len(surf.site_df))
        surf.sym_reduce()
        total_after = int(len(surf.site_df))
        conn = Counter(int(v) for v in surf.site_df["connectivity"].tolist())
        return {
            "status": "ok",
            "n_sites_raw": int(total_before),
            "n_sites_nonequivalent": int(total_after),
            "connectivity_counts": {str(k): int(v) for k, v in conn.items()},
        }
    except Exception as exc:
        return {"status": "runtime_error", "error": f"{type(exc).__name__}: {exc}"}


def _min_slab_ads_distance(slab: Atoms, ads: Atoms) -> float:
    spos = np.asarray(slab.get_positions(), dtype=float)
    apos = np.asarray(ads.get_positions(), dtype=float)
    d = apos[:, None, :] - spos[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    return float(np.min(dist))


def _placement_check_for_pair(
    slab: Atoms,
    molecule: Atoms,
    site_cache: dict[str, Any],
    *,
    min_accepted_distance: float,
) -> dict[str, Any]:
    primitives = site_cache["primitives"]
    surface_atom_ids = site_cache["surface_atom_ids"]
    if not primitives:
        return {"status": "zero_primitive", "n_poses": 0}
    sampler = PoseSampler(
        PoseSamplerConfig(
            n_rotations=1,
            n_azimuth=3,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.4,
            max_height=3.0,
            height_step=0.2,
            max_poses_per_site=1,
            random_seed=0,
        )
    )
    try:
        poses = sampler.sample(slab=slab, adsorbate=molecule, primitives=primitives, surface_atom_ids=surface_atom_ids)
    except Exception as exc:
        return {"status": "exception", "n_poses": 0, "error": f"{type(exc).__name__}: {exc}"}
    if not poses:
        retry = PoseSampler(
            PoseSamplerConfig(
                n_rotations=2,
                n_azimuth=4,
                n_shifts=2,
                shift_radius=0.2,
                min_height=1.8,
                max_height=4.2,
                height_step=0.15,
                clash_tau=0.70,
                site_contact_tolerance=0.35,
                max_poses_per_site=2,
                random_seed=17,
            )
        )
        try:
            poses = retry.sample(slab=slab, adsorbate=molecule, primitives=primitives, surface_atom_ids=surface_atom_ids)
        except Exception as exc:
            return {"status": "exception", "n_poses": 0, "error": f"{type(exc).__name__}: {exc}"}
    if not poses:
        slab_syms = set(slab.get_chemical_symbols())
        oxide_like = ("O" in slab_syms and len(slab_syms) >= 2)
        mono_like = len(molecule) <= 1
        far_min_h = 2.2
        far_max_h = 6.0
        far_clash_tau = 0.60
        far_contact_tol = 1.80
        if oxide_like:
            far_min_h = 2.8
            far_max_h = 6.5
            far_clash_tau = 0.75
            far_contact_tol = 1.20
        if mono_like:
            far_min_h = 2.0
            far_max_h = 6.5
            far_clash_tau = 0.45
            far_contact_tol = 3.00
        retry_far = PoseSampler(
            PoseSamplerConfig(
                n_rotations=2,
                n_azimuth=6,
                n_shifts=2,
                shift_radius=0.3,
                min_height=far_min_h,
                max_height=far_max_h,
                height_step=0.2,
                clash_tau=far_clash_tau,
                site_contact_tolerance=far_contact_tol,
                max_poses_per_site=3,
                random_seed=29,
            )
        )
        try:
            poses = retry_far.sample(slab=slab, adsorbate=molecule, primitives=primitives, surface_atom_ids=surface_atom_ids)
        except Exception as exc:
            return {"status": "exception", "n_poses": 0, "error": f"{type(exc).__name__}: {exc}"}
    if not poses:
        return {"status": "zero_pose", "n_poses": 0}
    dmins = [float(_min_slab_ads_distance(slab, p.atoms)) for p in poses]
    best = float(max(dmins))
    n_valid = int(sum(1 for d in dmins if d >= float(min_accepted_distance)))
    if n_valid <= 0:
        return {
            "status": "soft_clash",
            "n_poses": int(len(poses)),
            "n_valid_poses": int(n_valid),
            "min_interatomic_distance": float(best),
        }
    return {
        "status": "ok",
        "n_poses": int(len(poses)),
        "n_valid_poses": int(n_valid),
        "min_interatomic_distance": float(best),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_slab_suite()
    molecules = build_molecule_suite(max_atoms=int(args.max_molecule_atoms))
    slab_names = sorted(slabs.keys())
    mol_names = sorted(molecules.keys())
    if int(args.max_slabs) > 0:
        slab_names = slab_names[: int(args.max_slabs)]
    if int(args.max_molecules) > 0:
        mol_names = mol_names[: int(args.max_molecules)]

    slab_rows: list[dict[str, Any]] = []
    slab_cache: dict[str, dict[str, Any]] = {}
    for slab_name in slab_names:
        slab = slabs[slab_name]
        slab_dir = out_root / "slabs" / slab_name
        ours = _run_our_site_pipeline(slab, slab_dir)
        ours_counts = ours["basis_counts"]
        ase_ref = _ase_reference_counts(slab)
        ase_match = None
        if ase_ref.get("status") == "ok":
            exp = ase_ref["expected_mapped_counts"]
            obs = {
                "ontop": int(ours_counts.get("1c", 0)),
                "bridge": int(ours_counts.get("2c", 0)),
                "hollow": int(ours_counts.get("3c", 0) + ours_counts.get("4c", 0)),
            }
            ase_match = bool(obs == exp)
            ase_ref["observed_mapped_counts"] = obs
            ase_ref["strict_mapped_counts_match"] = ase_match
        auto = _autoadsorbate_site_counts(slab) if not bool(args.skip_autoadsorbate) else {"status": "skipped"}
        row = {
            "slab": slab_name,
            "ours": {k: v for k, v in ours.items() if k not in {"primitives", "surface_atom_ids"}},
            "ase_reference": ase_ref,
            "autoadsorbate": auto,
        }
        slab_rows.append(row)
        slab_cache[slab_name] = ours

    pair_rows: list[dict[str, Any]] = []
    stats = PlacementStats()
    for slab_name in slab_names:
        slab = slabs[slab_name]
        for mol_name in mol_names:
            stats.attempted += 1
            result = _placement_check_for_pair(
                slab,
                molecules[mol_name],
                slab_cache[slab_name],
                min_accepted_distance=float(args.min_accepted_distance),
            )
            status = str(result.get("status", "exception"))
            if status == "ok":
                stats.succeeded += 1
                stats.update_distance(float(result.get("min_interatomic_distance", 1.0e9)))
            elif status == "zero_primitive":
                stats.zero_primitive += 1
            elif status == "zero_pose":
                stats.zero_pose += 1
            elif status == "soft_clash":
                stats.soft_clash += 1
            else:
                stats.exception += 1
            pair_rows.append(
                {
                    "slab": slab_name,
                    "molecule": mol_name,
                    "n_atoms": int(len(molecules[mol_name])),
                    **result,
                }
            )

    if bool(args.save_sample_placements):
        sample_pairs = [
            ("Pt_fcc111", "CO"),
            ("Pt_fcc100", "H2O"),
            ("Pt_fcc211", "CH3OH"),
            ("Fe_bcc110", "C2H4"),
            ("Ru_hcp0001", "C6H6"),
            ("TiO2_110", "glycine_like"),
            ("Pt_fcc111_cluster_interface", "p_nitrobenzoic_acid_like"),
        ]
        sample_dir = out_root / "sample_placements"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for slab_name, mol_name in sample_pairs:
            if slab_name not in slabs or mol_name not in molecules or slab_name not in slab_cache:
                continue
            row = _placement_check_for_pair(
                slabs[slab_name],
                molecules[mol_name],
                slab_cache[slab_name],
                min_accepted_distance=float(args.min_accepted_distance),
            )
            if row.get("status") != "ok":
                continue
            sampler = PoseSampler(
                PoseSamplerConfig(
                    n_rotations=1,
                    n_azimuth=2,
                    n_shifts=1,
                    shift_radius=0.0,
                    min_height=1.5,
                    max_height=2.8,
                    height_step=0.2,
                    max_poses_per_site=1,
                    random_seed=0,
                )
            )
            poses = sampler.sample(
                slab=slabs[slab_name],
                adsorbate=molecules[mol_name],
                primitives=slab_cache[slab_name]["primitives"],
                surface_atom_ids=slab_cache[slab_name]["surface_atom_ids"],
            )
            frames = []
            for i, p in enumerate(poses[:8]):
                f = slabs[slab_name] + p.atoms
                f.info["pose_id"] = int(i)
                frames.append(f)
            if frames:
                write((sample_dir / f"{slab_name}__{mol_name}.extxyz").as_posix(), frames)

    payload = {
        "out_root": out_root.as_posix(),
        "n_slabs": int(len(slab_names)),
        "n_molecules": int(len(mol_names)),
        "n_pairs": int(len(pair_rows)),
        "placement_stats": stats.to_dict(),
        "slab_rows": slab_rows,
        "failed_pairs": [r for r in pair_rows if str(r.get("status")) != "ok"][:500],
    }
    out_path = out_root / "ase_autoadsorbate_crosscheck_summary.json"
    _write_json(out_path, payload)
    _write_json(out_root / "ase_autoadsorbate_pair_rows.json", pair_rows)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/physics_audit/ase_full_matrix")
    parser.add_argument("--max-slabs", type=int, default=0)
    parser.add_argument("--max-molecules", type=int, default=0)
    parser.add_argument("--max-molecule-atoms", type=int, default=40)
    parser.add_argument("--min-accepted-distance", type=float, default=0.70)
    parser.add_argument("--skip-autoadsorbate", action="store_true")
    parser.add_argument("--save-sample-placements", dest="save_sample_placements", action="store_true")
    parser.add_argument("--no-save-sample-placements", dest="save_sample_placements", action="store_false")
    parser.set_defaults(save_sample_placements=True)
    args = parser.parse_args()
    out_path = run(args)
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
