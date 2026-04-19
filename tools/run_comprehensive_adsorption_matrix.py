from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, bcc100, bcc110, bcc111, bulk, fcc100, fcc110, fcc111, fcc211, hcp0001, surface
from ase.cluster.icosahedron import Icosahedron
from ase.collections import g2
from ase.spacegroup import crystal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.conformer_md import ConformerMDSamplerConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.workflows import (
    AdsorptionWorkflowConfig,
    evaluate_adsorption_workflow_readiness,
    make_default_surface_preprocessor,
    plan_flex_sampling_budget,
    run_adsorption_workflow,
)
from tests.chemistry_cases import get_test_adsorbate_cases
from tools.run_autoresearch_artifact_suite import resolve_mace_model_path, runtime_manifest


def _center_adsorbate(a: Atoms) -> Atoms:
    out = a.copy()
    pos = np.asarray(out.get_positions(), dtype=float)
    pos -= np.mean(pos, axis=0, keepdims=True)
    out.set_positions(pos)
    return out


def build_adsorbates() -> dict[str, Atoms]:
    ads = dict(get_test_adsorbate_cases())
    chosen = []
    for name in g2.names:
        try:
            a = g2[name].copy()
        except Exception:
            continue
        n = len(a)
        if n < 6 or n > 28:
            continue
        syms = set(a.get_chemical_symbols())
        hetero = sorted([s for s in syms if s not in {"H", "C"}])
        score = n + 2 * len(hetero)
        if len(hetero) == 0:
            continue
        chosen.append((score, name, _center_adsorbate(a)))
    chosen = sorted(chosen, key=lambda x: (-x[0], x[1]))
    for _, name, a in chosen[:30]:
        key = f"g2_{name}"
        ads[key] = a
    return ads


def _replace_top_layer_atoms(base: Atoms, from_symbol: str, to_symbol: str, frac: float) -> Atoms:
    slab = base.copy()
    z = np.asarray(slab.get_positions(), dtype=float)[:, 2]
    z_top = float(np.max(z))
    top = [i for i, zi in enumerate(z) if (z_top - zi) < 1.0]
    n_swap = max(1, int(round(len(top) * float(frac))))
    for i in top[:n_swap]:
        if slab[i].symbol == from_symbol:
            slab[i].symbol = to_symbol
    return slab


def _with_surface_vacancy(base: Atoms) -> Atoms:
    slab = base.copy()
    z = np.asarray(slab.get_positions(), dtype=float)[:, 2]
    top_idx = int(np.argmax(z))
    del slab[top_idx]
    return slab


def _with_surface_adatom(base: Atoms, symbol: str = "Pt") -> Atoms:
    slab = base.copy()
    z = np.asarray(slab.get_positions(), dtype=float)[:, 2]
    xy = np.asarray(slab.get_positions(), dtype=float)[:, :2]
    idx = int(np.argmax(z))
    add_adsorbate(slab, symbol, 1.8, position=xy[idx])
    return slab


def _with_cluster_interface(base: Atoms, symbol: str = "Pt") -> Atoms:
    slab = base.copy()
    c = Icosahedron(symbol, 1).copy()
    pos = np.asarray(c.get_positions(), dtype=float)
    pos -= np.mean(pos, axis=0, keepdims=True)
    s_pos = np.asarray(slab.get_positions(), dtype=float)
    top = float(np.max(s_pos[:, 2]))
    center_xy = np.mean(s_pos[:, :2], axis=0)
    pos[:, 0] += center_xy[0]
    pos[:, 1] += center_xy[1]
    pos[:, 2] += top + 2.2
    c.set_positions(pos)
    return slab + c


def build_slabs() -> dict[str, Atoms]:
    slabs: dict[str, Atoms] = {}
    # Pure metal Miller surfaces
    slabs["Pt_fcc111"] = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
    slabs["Pt_fcc100"] = fcc100("Pt", size=(4, 4, 4), vacuum=12.0)
    slabs["Pt_fcc110"] = fcc110("Pt", size=(4, 4, 4), vacuum=12.0)
    slabs["Pt_fcc211"] = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
    slabs["Cu_fcc111"] = fcc111("Cu", size=(4, 4, 4), vacuum=12.0)
    slabs["Ni_fcc111"] = fcc111("Ni", size=(4, 4, 4), vacuum=12.0)
    slabs["Fe_bcc100"] = bcc100("Fe", size=(4, 4, 4), vacuum=12.0)
    slabs["Fe_bcc110"] = bcc110("Fe", size=(4, 4, 4), vacuum=12.0)
    slabs["Fe_bcc111"] = bcc111("Fe", size=(4, 4, 4), vacuum=12.0)
    slabs["Ti_hcp0001"] = hcp0001("Ti", size=(4, 4, 4), vacuum=12.0)
    # Oxides
    rutile = crystal(
        symbols=["Ti", "O"],
        basis=[(0.0, 0.0, 0.0), (0.305, 0.305, 0.0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )
    slabs["TiO2_110"] = surface(rutile, (1, 1, 0), layers=6, vacuum=12.0).repeat((2, 2, 1))
    rocksalt = bulk("MgO", "rocksalt", a=4.213)
    slabs["MgO_100"] = surface(rocksalt, (1, 0, 0), layers=6, vacuum=12.0).repeat((2, 2, 1))
    # Alloy / defect / interface
    slabs["CuNi_fcc111_alloy"] = _replace_top_layer_atoms(fcc111("Cu", size=(4, 4, 4), vacuum=12.0), "Cu", "Ni", frac=0.25)
    slabs["Pt_fcc111_vacancy"] = _with_surface_vacancy(fcc111("Pt", size=(4, 4, 4), vacuum=12.0))
    slabs["Pt_fcc111_adatom"] = _with_surface_adatom(fcc111("Pt", size=(4, 4, 4), vacuum=12.0), symbol="Pt")
    slabs["Pt_fcc111_cluster_interface"] = _with_cluster_interface(fcc111("Pt", size=(4, 4, 4), vacuum=12.0), symbol="Pt")
    return slabs


def is_flexible_adsorbate(a: Atoms) -> bool:
    n = len(a)
    syms = set(a.get_chemical_symbols())
    if n >= 10:
        return True
    if "N" in syms or "O" in syms:
        return n >= 8
    return False


def build_workflow_config(
    *,
    work_dir: Path,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
    run_conformer_search: bool,
    budget=None,
) -> AdsorptionWorkflowConfig:
    model_key = ("" if mace_model_path is None else Path(mace_model_path).name.lower())
    head_name = "omat_pbe" if "omat" in model_key else "Default"
    conformer_cfg = ConformerMDSamplerConfig()
    conformer_cfg.md.temperature_k = 450.0
    conformer_cfg.md.time_ps = (10.0 if budget is None else float(budget.md_time_ps))
    conformer_cfg.md.step_fs = 1.0
    conformer_cfg.md.dump_fs = 50.0
    conformer_cfg.md.n_runs = (2 if budget is None else int(budget.md_runs))
    conformer_cfg.descriptor.backend = "mace"
    conformer_cfg.descriptor.mace.model_path = mace_model_path
    conformer_cfg.descriptor.mace.device = str(mace_device)
    conformer_cfg.descriptor.mace.dtype = str(mace_dtype)
    conformer_cfg.descriptor.mace.head_name = head_name
    conformer_cfg.relax.backend = "mace_energy"
    conformer_cfg.relax.mace.model_path = mace_model_path
    conformer_cfg.relax.mace.device = str(mace_device)
    conformer_cfg.relax.mace.dtype = str(mace_dtype)
    conformer_cfg.relax.mace.head_name = head_name
    conformer_cfg.selection.preselect_k = (96 if budget is None else int(budget.preselect_k))
    conformer_cfg.selection.energy_window_ev = 0.30
    conformer_cfg.selection.rmsd_threshold = 0.08
    conformer_cfg.selection.fps_convergence_enable = True
    conformer_cfg.selection.fps_convergence_grid_bins = 16
    conformer_cfg.selection.fps_convergence_min_rounds = (4 if budget is None else int(budget.fps_rounds))
    conformer_cfg.selection.fps_round_size = (None if budget is None else int(budget.fps_round_size))
    conformer_cfg.selection.fps_convergence_patience = 3
    conformer_cfg.output.save_all_frames = True
    conformer_cfg.output.work_dir = work_dir / "conformer_md"
    return AdsorptionWorkflowConfig(
        work_dir=work_dir,
        surface_preprocessor=make_default_surface_preprocessor(),
        pose_sampler_config=PoseSamplerConfig(
            n_rotations=6,
            n_azimuth=12,
            n_shifts=3,
            shift_radius=0.25,
            min_height=1.4,
            max_height=3.4,
            height_step=0.15,
            max_poses_per_site=6,
            random_seed=0,
        ),
        basin_config=BasinConfig(
            relax_maxf=0.1,
            relax_steps=80,
            energy_window_ev=2.5,
            dedup_metric="mace_node_l2",
            mace_node_l2_threshold=2.0,
            desorption_min_bonds=0,
            work_dir=None,
            mace_model_path=mace_model_path,
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            mace_max_edges_per_batch=20000,
            mace_layers_to_keep=-1,
            mace_head_name=head_name,
        ),
        run_conformer_search=bool(run_conformer_search),
        conformer_config=conformer_cfg,
        conformer_job_name="conformer_search",
        max_primitives=None,
        max_selected_primitives=36,
        save_basin_dictionary=True,
        save_basin_ablation=True,
        basin_ablation_metrics=("signature_only", "rmsd", "mace_node_l2"),
        save_site_visualizations=True,
        save_raw_site_dictionary=True,
        save_selected_site_dictionary=True,
        primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.20),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/comprehensive_omat_cuda_fp32")
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--max-standard-cases", type=int, default=0)
    parser.add_argument("--max-flex-cases", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path, model_source = resolve_mace_model_path(args.mace_model_path)
    device_requested = str(args.mace_device)
    device_effective = str(device_requested)
    if str(device_requested).lower().startswith("cuda"):
        ok = False
        try:
            import torch  # type: ignore

            ok = bool(torch.cuda.is_available())
        except Exception:
            ok = False
        if not ok and bool(args.require_cuda):
            raise RuntimeError("CUDA is required for MACE, but torch.cuda.is_available() is False.")
    runtime_manifest(
        mace_model_path=model_path,
        model_source=model_source,
        mace_device=device_requested,
        mace_device_effective=device_effective,
        mace_dtype=str(args.mace_dtype),
        out_root=out_root,
    )

    slabs = build_slabs()
    adsorbates = build_adsorbates()
    slab_names = sorted(slabs.keys())
    ads_names = sorted(adsorbates.keys())

    standard_pairs: list[tuple[str, str]] = []
    flexible_pairs: list[tuple[str, str]] = []
    for i, slab_name in enumerate(slab_names):
        for j, ads_name in enumerate(ads_names):
            if (i + j) % 3 != 0:
                continue
            ads = adsorbates[ads_name]
            if is_flexible_adsorbate(ads):
                flexible_pairs.append((slab_name, ads_name))
            else:
                standard_pairs.append((slab_name, ads_name))

    if int(args.max_standard_cases) > 0:
        standard_pairs = standard_pairs[: int(args.max_standard_cases)]
    if int(args.max_flex_cases) > 0:
        flexible_pairs = flexible_pairs[: int(args.max_flex_cases)]

    rows = []
    model_key = ("" if model_path is None else Path(model_path).name.lower())
    head_name = "omat_pbe" if "omat" in model_key else ("omol" if "omol" in model_key else None)
    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=device_effective,
            dtype=str(args.mace_dtype),
            max_edges_per_batch=20000,
            head_name=head_name,
            strict=True,
        )
    )
    for category, pairs, run_conformer in (
        ("standard", standard_pairs, False),
        ("flexible_conformer", flexible_pairs, True),
    ):
        for slab_name, ads_name in pairs:
            case_dir = out_root / category / slab_name / ads_name
            budget = plan_flex_sampling_budget(
                adsorbates[ads_name],
                n_surface_atoms=len(slabs[slab_name]),
                n_site_primitives=36,
            )
            cfg = build_workflow_config(
                work_dir=case_dir,
                mace_model_path=model_path,
                mace_device=device_effective,
                mace_dtype=str(args.mace_dtype),
                run_conformer_search=bool(run_conformer and budget.run_conformer_search),
                budget=budget,
            )
            result = run_adsorption_workflow(
                slab=slabs[slab_name],
                adsorbate=adsorbates[ads_name],
                config=cfg,
                basin_relax_backend=relax_backend,
            )
            ready = evaluate_adsorption_workflow_readiness(result)
            rows.append(
                {
                    "category": category,
                    "slab": slab_name,
                    "adsorbate": ads_name,
                    "run_conformer_search": bool(run_conformer and budget.run_conformer_search),
                    "flex_score": float(budget.score),
                    "flex_budget": dict(budget.rationale),
                    "n_surface_atoms": int(result.summary["n_surface_atoms"]),
                    "n_raw_primitives": int(result.summary["n_raw_primitives"]),
                    "n_selected_primitives": int(result.summary["n_primitives"]),
                    "n_basis_primitives": int(result.summary["n_basis_primitives"]),
                    "n_conformers": int(result.summary["n_conformers"]),
                    "n_pose_frames": int(result.summary["n_pose_frames"]),
                    "n_basins": int(result.summary["n_basins"]),
                    "n_nodes": int(result.summary["n_nodes"]),
                    "paper_readiness_score": int(ready.score),
                    "paper_readiness_max_score": int(ready.max_score),
                    "work_dir": case_dir.as_posix(),
                }
            )

    payload = {
        "out_root": out_root.as_posix(),
        "mace_model_path": model_path,
        "mace_model_source": model_source,
        "mace_device_requested": device_requested,
        "mace_device_effective": device_effective,
        "mace_dtype": str(args.mace_dtype),
        "n_slabs": len(slabs),
        "n_adsorbates": len(adsorbates),
        "n_standard_cases": len(standard_pairs),
        "n_flexible_cases": len(flexible_pairs),
        "rows": rows,
    }
    (out_root / "matrix_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_root": payload["out_root"],
                "n_slabs": payload["n_slabs"],
                "n_adsorbates": payload["n_adsorbates"],
                "n_standard_cases": payload["n_standard_cases"],
                "n_flexible_cases": payload["n_flexible_cases"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
