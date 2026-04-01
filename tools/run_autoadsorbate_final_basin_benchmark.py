from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc111, fcc211
from ase.io import write

from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.basin.dedup import kabsch_rmsd
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, run_adsorption_workflow
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from autoadsorbate import Surface, get_marked_smiles
from autoadsorbate.Smile import atoms_from_smile, create_adsorbates
from autoadsorbate.Surf import _place_adsorbate


def _make_relax_backend() -> MACEBatchRelaxBackend:
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


def _make_workflow_config(work_dir: Path) -> AdsorptionWorkflowConfig:
    return AdsorptionWorkflowConfig(
        work_dir=work_dir,
        surface_preprocessor=SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=0.55),
            fallback_detector=VoxelFloodDetector(spacing=0.75),
            target_surface_fraction=None,
            target_count_mode="off",
        ),
        pose_sampler_config=PoseSamplerConfig(
            n_rotations=4,
            n_azimuth=8,
            n_shifts=2,
            shift_radius=0.20,
            min_height=1.4,
            max_height=3.6,
            height_step=0.15,
            max_poses_per_site=4,
            random_seed=0,
        ),
        basin_config=BasinConfig(
            relax_maxf=0.1,
            relax_steps=80,
            energy_window_ev=2.5,
            dedup_metric="rmsd",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            desorption_min_bonds=0,
            work_dir=None,
        ),
        max_primitives=None,
        max_selected_primitives=24,
        save_basin_dictionary=True,
        save_basin_ablation=False,
        save_site_visualizations=True,
        save_raw_site_dictionary=True,
        save_selected_site_dictionary=True,
        primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.20),
    )


def _unified_basin_config(work_dir: Path) -> BasinConfig:
    return BasinConfig(
        relax_maxf=0.1,
        relax_steps=80,
        energy_window_ev=2.5,
        dedup_metric="rmsd",
        dedup_cluster_method="hierarchical",
        rmsd_threshold=0.10,
        desorption_min_bonds=0,
        work_dir=work_dir,
    )


def _build_cases() -> list[dict[str, Any]]:
    return [
        {"name": "Pt111_methanol", "slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "smiles": "CO"},
        {"name": "Pt100_methanol", "slab": fcc100("Pt", size=(4, 4, 4), vacuum=12.0), "smiles": "CO"},
        {"name": "Pt111_dimethyl_ether", "slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "smiles": "COC"},
        {"name": "Pt211_ethanol", "slab": fcc211("Pt", size=(6, 4, 4), vacuum=12.0), "smiles": "CCO"},
        {"name": "Pt111_methylamine", "slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "smiles": "CN"},
    ]


def _build_autoadsorbate_frames(slab: Atoms, smiles: str, out_dir: Path) -> tuple[list[Atoms], dict[str, Any]]:
    surf = Surface(slab.copy(), mode="slab", precision=0.35, touch_sphere_size=3.0)
    slab_n = int(len(slab))
    surf.sym_reduce()
    top_sites = []
    for i in range(len(surf.site_df)):
        row = surf.site_df.iloc[i].to_dict()
        if int(row["connectivity"]) == 1:
            top_sites.append(row)
    marked_smiles = list(get_marked_smiles([smiles], attack_atoms=["O", "N"]))
    frames: list[Atoms] = []
    accepted = 0
    attempted = 0
    site_counter: Counter[str] = Counter()
    for marked in marked_smiles:
        try:
            adsorbates = create_adsorbates(marked, conf_no=1, max_tries=3)
        except Exception:
            continue
        for ads in adsorbates:
            for site in top_sites:
                attempted += 1
                try:
                    placed = _place_adsorbate(
                        slab.copy(),
                        ads.copy(),
                        surf_nvector=np.asarray(site["n_vector"], dtype=float).copy(),
                        site=np.asarray(site["coordinates"], dtype=float),
                        surf_hvector=np.asarray(site["h_vector"], dtype=float).copy(),
                    )
                except Exception:
                    continue
                ads_n = int(len(placed) - slab_n)
                spos = np.asarray(placed.get_positions(), dtype=float)[:slab_n]
                apos = np.asarray(placed.get_positions(), dtype=float)[slab_n : slab_n + ads_n]
                dmat = np.linalg.norm(apos[:, None, :] - spos[None, :, :], axis=2)
                mfd = float(np.min(dmat)) if dmat.size else 0.0
                if not np.isfinite(mfd) or mfd < 1.0:
                    continue
                placed.info["generator"] = "autoadsorbate"
                placed.info["base_smiles"] = str(smiles)
                placed.info["marked_smiles"] = str(marked)
                placed.info["site_connectivity"] = int(site["connectivity"])
                frames.append(placed)
                accepted += 1
                site_counter[str(site["connectivity"])] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    if frames:
        write((out_dir / "autoadsorbate_candidates.extxyz").as_posix(), frames)
    site_dict = surf.site_df.to_dict(orient="list")
    (out_dir / "autoadsorbate_site_dict.json").write_text(json.dumps(site_dict, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return frames, {
        "n_marked_smiles": int(len(marked_smiles)),
        "marked_smiles": [str(x) for x in marked_smiles],
        "n_top_sites": int(len(top_sites)),
        "n_attempted": int(attempted),
        "n_accepted": int(accepted),
        "accepted_site_connectivity_counts": dict(site_counter),
    }


def _basin_overlap(ours_frames: list[Atoms], other_frames: list[Atoms], slab_n: int, rmsd_threshold: float = 0.20) -> dict[str, Any]:
    matched = 0
    matches = []
    for j, b in enumerate(other_frames):
        b_pos = np.asarray(b.get_positions(), dtype=float)[slab_n:]
        found = None
        for i, a in enumerate(ours_frames):
            if str(a.info.get("signature", "")) != str(b.info.get("signature", "")):
                continue
            a_pos = np.asarray(a.get_positions(), dtype=float)[slab_n:]
            d = float(kabsch_rmsd(a_pos, b_pos))
            if np.isfinite(d) and d <= float(rmsd_threshold):
                found = {"ours_index": int(i), "other_index": int(j), "rmsd": float(d), "signature": str(a.info.get("signature", ""))}
                break
        if found is not None:
            matched += 1
            matches.append(found)
    return {
        "matched_other_basins": int(matched),
        "n_other_basins": int(len(other_frames)),
        "n_ours_basins": int(len(ours_frames)),
        "other_recall_vs_ours": float(matched / max(1, len(other_frames))),
        "coverage_of_other_by_ours": float(matched / max(1, len(other_frames))),
        "matches": matches,
    }


def _run_autoadsorbate_basins(
    slab: Atoms,
    smiles: str,
    work_dir: Path,
    relax_backend: MACEBatchRelaxBackend,
) -> tuple[list[Atoms], dict[str, Any]]:
    base_ads = atoms_from_smile(smiles)
    frames, gen_meta = _build_autoadsorbate_frames(slab=slab, smiles=smiles, out_dir=work_dir)
    if not frames:
        return [], {"generator": dict(gen_meta), "basin_summary": {"n_input": 0, "n_basins": 0}}
    basin_cfg = _unified_basin_config(work_dir=work_dir / "basin_work")
    result = BasinBuilder(config=basin_cfg, relax_backend=relax_backend).build(
        frames=frames,
        slab_ref=slab,
        adsorbate_ref=base_ads,
        slab_n=len(slab),
        normal_axis=2,
    )
    basin_frames = []
    for basin in result.basins:
        a = basin.atoms.copy()
        a.info["basin_id"] = int(basin.basin_id)
        a.info["signature"] = str(basin.signature)
        a.info["energy_ev"] = float(basin.energy_ev)
        basin_frames.append(a)
    if basin_frames:
        write((work_dir / "autoadsorbate_basins.extxyz").as_posix(), basin_frames)
    rejected_reason_counts = dict(Counter(str(r.reason) for r in result.rejected))
    (work_dir / "autoadsorbate_basin_summary.json").write_text(
        json.dumps(
            {
                "generator": gen_meta,
                "basin_summary": result.summary,
                "rejected_reason_counts": rejected_reason_counts,
                "rejected": [{"candidate_id": int(r.candidate_id), "reason": str(r.reason), "metrics": dict(r.metrics)} for r in result.rejected],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return basin_frames, {"generator": dict(gen_meta), "basin_summary": dict(result.summary), "rejected_reason_counts": rejected_reason_counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/final_basin_crosslib")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    relax_backend = _make_relax_backend()

    rows = []
    cases = _build_cases()
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]
    for case in cases:
        case_dir = out_root / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        slab = case["slab"]
        smiles = str(case["smiles"])
        ads = atoms_from_smile(smiles)

        ours_cfg = _make_workflow_config(case_dir / "ours")
        ours_result = run_adsorption_workflow(
            slab=slab,
            adsorbate=ads,
            config=ours_cfg,
            basin_relax_backend=relax_backend,
        )
        ours_basins = []
        if ours_result.artifacts.get("basins_extxyz"):
            try:
                from ase.io import read

                ours_basins = list(read(ours_result.artifacts["basins_extxyz"], index=":"))
            except Exception:
                ours_basins = []

        auto_basins, auto_meta = _run_autoadsorbate_basins(
            slab=slab,
            smiles=smiles,
            work_dir=case_dir / "autoadsorbate",
            relax_backend=relax_backend,
        )
        overlap = _basin_overlap(ours_frames=ours_basins, other_frames=auto_basins, slab_n=len(slab), rmsd_threshold=0.20)
        row = {
            "case": case["name"],
            "smiles": smiles,
            "ours": {
                "n_surface_atoms": int(ours_result.summary["n_surface_atoms"]),
                "n_basis_primitives": int(ours_result.summary["n_basis_primitives"]),
                "n_pose_frames": int(ours_result.summary["n_pose_frames"]),
                "n_basins": int(ours_result.summary["n_basins"]),
                "work_dir": (case_dir / "ours").as_posix(),
            },
            "autoadsorbate": {
                **auto_meta,
                "n_basins": int(len(auto_basins)),
                "work_dir": (case_dir / "autoadsorbate").as_posix(),
            },
            "overlap": overlap,
        }
        rows.append(row)

    summary = {
        "out_root": out_root.as_posix(),
        "mace_model_path": "/root/.cache/mace/mace-omat-0-small.model",
        "rows": rows,
        "notes": {
            "autoadsorbate_scope": "This benchmark uses AutoAdsorbate marked-SMILES O/N monodentate pathway and its shrinkwrap top-site generation, then relaxes all candidates with the same MACE backend used by our workflow.",
            "dockonsurf_status": "DockOnSurf remains a next-stage anchor-aware baseline for flexible adsorption screening and is not included in this benchmark artifact yet.",
        },
    }
    out_path = out_root / "autoadsorbate_final_basin_benchmark.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    slab_n = len(slab)
