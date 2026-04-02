from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc100, fcc110, fcc111, molecule
from ase.io import read, write

from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.basin.dedup import kabsch_rmsd
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.relax.backends import IdentityRelaxBackend, MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, run_adsorption_workflow


def _make_relax_backend(kind: str) -> object:
    if str(kind).strip().lower() == "fake":
        return IdentityRelaxBackend()
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path="/root/.cache/mace/mace-omat-0-small.model",
            device="cuda",
            dtype="float32",
            max_edges_per_batch=20000,
            head_name="omat_pbe",
            strict=True,
        )
    )


def _workflow_config(work_dir: Path, ads_name: str, *, placement_mode: str = "anchor_free") -> AdsorptionWorkflowConfig:
    if ads_name == "H":
        pose_cfg = PoseSamplerConfig(
            placement_mode=str(placement_mode),
            n_rotations=1,
            n_azimuth=1,
            n_shifts=1,
            shift_radius=0.0,
            min_height=0.8,
            max_height=2.2,
            height_step=0.10,
            max_poses_per_site=3,
            random_seed=0,
        )
    else:
        pose_cfg = PoseSamplerConfig(
            placement_mode=str(placement_mode),
            n_rotations=4,
            n_azimuth=8,
            n_shifts=2,
            shift_radius=0.15,
            min_height=1.2,
            max_height=3.2,
            height_step=0.10,
            max_poses_per_site=4,
            random_seed=0,
        )
    return AdsorptionWorkflowConfig(
        work_dir=work_dir,
        surface_preprocessor=SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=0.55),
            fallback_detector=VoxelFloodDetector(spacing=0.75),
            target_surface_fraction=None,
            target_count_mode="off",
        ),
        pose_sampler_config=pose_cfg,
        basin_config=BasinConfig(
            relax_maxf=0.1,
            relax_steps=80,
            energy_window_ev=2.5,
            dedup_metric="rmsd",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            desorption_min_bonds=1,
            work_dir=None,
        ),
        max_selected_primitives=24,
        save_basin_dictionary=True,
        save_site_visualizations=True,
        save_raw_site_dictionary=True,
        save_selected_site_dictionary=True,
        primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.20),
    )


def _manual_basin_config(work_dir: Path) -> BasinConfig:
    return BasinConfig(
        relax_maxf=0.1,
        relax_steps=80,
        energy_window_ev=2.5,
        dedup_metric="rmsd",
        dedup_cluster_method="hierarchical",
        rmsd_threshold=0.10,
        desorption_min_bonds=1,
        work_dir=work_dir,
    )


def _make_adsorbate(name: str) -> Atoms:
    if name == "H":
        return Atoms("H", positions=[[0.0, 0.0, 0.0]])
    if name == "CO":
        co = molecule("CO")
        if co[0].symbol != "C":
            co = co[[1, 0]]
        return co
    raise ValueError(name)


def _build_cases() -> list[dict[str, Any]]:
    slabs = {
        "Pt111": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
        "Pt100": fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
        "Pt110": fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
    }
    out = []
    for slab_name, slab in slabs.items():
        for ads_name in ("H", "CO"):
            out.append({"name": f"{slab_name}_{ads_name}", "slab": slab, "ads_name": ads_name})
    return out


def _manual_height(ads_name: str, site_name: str) -> float:
    if ads_name == "H":
        if site_name in {"hollow", "fcc", "hcp"}:
            return 0.9
        if "bridge" in site_name:
            return 1.0
        return 1.1
    if site_name in {"hollow", "fcc", "hcp"}:
        return 1.6
    if "bridge" in site_name:
        return 1.75
    return 1.85


def _manual_frames_for_ase_sites(slab: Atoms, ads_name: str) -> tuple[list[Atoms], list[dict[str, Any]]]:
    info = slab.info.get("adsorbate_info", {})
    sites = info.get("sites", {}) if isinstance(info, dict) else {}
    frames = []
    meta = []
    ads = _make_adsorbate(ads_name)
    for site_name in sites.keys():
        placed = slab.copy()
        add_adsorbate(placed, ads.copy(), _manual_height(ads_name, str(site_name).lower()), str(site_name))
        placed.info["generator"] = "ase_manual"
        placed.info["site_name"] = str(site_name)
        placed.info["site_label"] = str(site_name).lower()
        frames.append(placed)
        meta.append({"site_name": str(site_name)})
    return frames, meta


def _load_basins(extxyz_path: str | None) -> list[Atoms]:
    if not extxyz_path:
        return []
    try:
        return list(read(extxyz_path, index=":"))
    except Exception:
        return []


def _overlap(manual_basins: list[Atoms], ours_basins: list[Atoms], slab_n: int, rmsd_threshold: float = 0.20) -> dict[str, Any]:
    matched = 0
    matches = []
    for j, b in enumerate(manual_basins):
        b_pos = np.asarray(b.get_positions(), dtype=float)[slab_n:]
        found = None
        for i, a in enumerate(ours_basins):
            sig_a = str(a.info.get("signature", ""))
            sig_b = str(b.info.get("signature", ""))
            if sig_a != sig_b:
                continue
            a_pos = np.asarray(a.get_positions(), dtype=float)[slab_n:]
            d = float(kabsch_rmsd(a_pos, b_pos))
            if np.isfinite(d) and d <= float(rmsd_threshold):
                found = {"manual_index": int(j), "ours_index": int(i), "signature": sig_a, "rmsd": float(d)}
                break
        if found is not None:
            matched += 1
            matches.append(found)
    return {
        "matched_manual_basins": int(matched),
        "n_manual_basins": int(len(manual_basins)),
        "n_ours_basins": int(len(ours_basins)),
        "manual_recall_by_ours": float(matched / max(1, len(manual_basins))),
        "matches": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/simple_case_validation")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--relax-backend", type=str, default="mace", choices=["mace", "fake"])
    parser.add_argument("--placement-mode", type=str, default="anchor_free", choices=["anchor_free", "anchor_aware"])
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    relax_backend = _make_relax_backend(args.relax_backend)
    cases = _build_cases()
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    rows = []
    for case in cases:
        case_dir = out_root / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        slab = case["slab"]
        ads_name = str(case["ads_name"])
        ads = _make_adsorbate(ads_name)

        ours_cfg = _workflow_config(case_dir / "ours", ads_name=ads_name, placement_mode=str(args.placement_mode))
        ours_result = run_adsorption_workflow(
            slab=slab,
            adsorbate=ads,
            config=ours_cfg,
            basin_relax_backend=relax_backend,
        )
        ours_basins = _load_basins(ours_result.artifacts.get("basins_extxyz"))

        manual_frames, manual_meta = _manual_frames_for_ase_sites(slab=slab, ads_name=ads_name)
        manual_cfg = _manual_basin_config(case_dir / "ase_manual" / "basin_work")
        manual_result = BasinBuilder(config=manual_cfg, relax_backend=relax_backend).build(
            frames=manual_frames,
            slab_ref=slab,
            adsorbate_ref=ads,
            slab_n=len(slab),
            normal_axis=2,
        )
        manual_basins = []
        for basin in manual_result.basins:
            a = basin.atoms.copy()
            a.info["signature"] = str(basin.signature)
            a.info["basin_id"] = int(basin.basin_id)
            a.info["energy_ev"] = float(basin.energy_ev)
            manual_basins.append(a)
        if manual_basins:
            write((case_dir / "ase_manual" / "manual_basins.extxyz").as_posix(), manual_basins)
        (case_dir / "ase_manual" / "manual_basin_summary.json").write_text(
            json.dumps(
                {
                    "manual_sites": manual_meta,
                    "summary": manual_result.summary,
                    "rejected": [{"candidate_id": int(r.candidate_id), "reason": str(r.reason), "metrics": dict(r.metrics)} for r in manual_result.rejected],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        overlap = _overlap(manual_basins=manual_basins, ours_basins=ours_basins, slab_n=len(slab), rmsd_threshold=0.20)
        rows.append(
            {
                "case": case["name"],
                "adsorbate": ads_name,
                "ours": {
                    "n_basis_primitives": int(ours_result.summary["n_basis_primitives"]),
                    "n_pose_frames": int(ours_result.summary["n_pose_frames"]),
                    "n_basins": int(ours_result.summary["n_basins"]),
                    "placement_mode": str(args.placement_mode),
                    "work_dir": (case_dir / "ours").as_posix(),
                },
                "ase_manual": {
                    "n_input_sites": int(len(manual_frames)),
                    "n_basins": int(len(manual_basins)),
                    "rejected_reason_counts": dict({str(k): int(v) for k, v in Counter(str(r.reason) for r in manual_result.rejected).items()}),
                    "work_dir": (case_dir / "ase_manual").as_posix(),
                },
                "overlap": overlap,
            }
        )

    payload = {
        "out_root": out_root.as_posix(),
        "relax_backend": str(args.relax_backend),
        "placement_mode": str(args.placement_mode),
        "mace_model_path": ("/root/.cache/mace/mace-omat-0-small.model" if str(args.relax_backend) == "mace" else None),
        "rows": rows,
    }
    out_path = out_root / "simple_case_validation.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
