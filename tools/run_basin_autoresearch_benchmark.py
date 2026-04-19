from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

from ase.build import bcc110, bulk, fcc100, fcc111, fcc211, surface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.workflows import (
    AdsorptionWorkflowConfig,
    evaluate_adsorption_workflow_readiness,
    make_default_surface_preprocessor,
    run_adsorption_workflow,
)
from tests.chemistry_cases import get_test_adsorbate_cases


def build_slabs():
    return {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=10.0),
        "bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=10.0),
        "cu321": surface(bulk("Cu", "fcc", a=3.6, cubic=True), (3, 2, 1), layers=4, vacuum=10.0).repeat((2, 2, 1)),
    }


def infer_head_name(model_path: str) -> str | None:
    lower = Path(model_path).name.lower()
    if "omat" in lower:
        return "omat_pbe"
    if "omol" in lower:
        return "omol"
    return None


def ratio_score(n_basins: int, n_pose: int) -> float:
    ratio = float(n_basins) / float(max(1, n_pose))
    return max(0.0, 1.0 - abs(ratio - 0.65) / 0.65)


def runtime_score(runtime_sec: float, target_sec: float = 180.0) -> float:
    return max(0.0, 1.0 - float(runtime_sec) / float(target_sec))


def gap_score(n_mace: int, n_rmsd: int, n_pose: int) -> float:
    gap = abs(float(n_mace) - float(n_rmsd)) / float(max(1, n_pose))
    return max(0.0, 1.0 - 5.0 * gap)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/benchmark")
    parser.add_argument("--mace-model-path", type=str, required=True)
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--max-selected-primitives", type=int, default=24)
    parser.add_argument("--pose-n-rotations", type=int, default=4)
    parser.add_argument("--pose-n-azimuth", type=int, default=8)
    parser.add_argument("--pose-n-shifts", type=int, default=2)
    parser.add_argument("--pose-max-per-site", type=int, default=4)
    parser.add_argument("--relax-maxf", type=float, default=0.10)
    parser.add_argument("--relax-steps", type=int, default=80)
    parser.add_argument("--energy-window-ev", type=float, default=2.0)
    parser.add_argument("--mace-node-l2-threshold", type=float, default=2.0)
    parser.add_argument("--desorption-min-bonds", type=int, default=0)
    parser.add_argument("--surface-reconstruction-max-disp", type=float, default=0.50)
    parser.add_argument("--dissociation-allow-bond-change", action="store_true")
    parser.add_argument("--burial-margin", type=float, default=0.30)
    args = parser.parse_args()

    if str(args.mace_device).lower().startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for benchmark but unavailable.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_slabs()
    ads = get_test_adsorbate_cases()
    cases = [
        ("fcc111", "CO"),
        ("fcc100", "H2O"),
        ("fcc211", "CH3OH"),
        ("bcc110", "C2H4"),
        ("cu321", "C6H6"),
    ]

    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=str(args.mace_model_path),
            device=str(args.mace_device),
            dtype=str(args.mace_dtype),
            max_edges_per_batch=20000,
            head_name=infer_head_name(str(args.mace_model_path)),
            strict=True,
        )
    )

    rows = []
    for slab_name, ads_name in cases:
        work_dir = out_root / slab_name / ads_name
        cfg = AdsorptionWorkflowConfig(
            work_dir=work_dir,
            surface_preprocessor=make_default_surface_preprocessor(),
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=int(args.pose_n_rotations),
                n_azimuth=int(args.pose_n_azimuth),
                n_shifts=int(args.pose_n_shifts),
                shift_radius=0.2,
                min_height=1.5,
                max_height=3.0,
                height_step=0.2,
                max_poses_per_site=int(args.pose_max_per_site),
                random_seed=0,
            ),
            basin_config=BasinConfig(
                relax_maxf=float(args.relax_maxf),
                relax_steps=int(args.relax_steps),
                energy_window_ev=float(args.energy_window_ev),
                dedup_metric="mace_node_l2",
                mace_node_l2_threshold=float(args.mace_node_l2_threshold),
                desorption_min_bonds=int(args.desorption_min_bonds),
                surface_reconstruction_max_disp=float(args.surface_reconstruction_max_disp),
                dissociation_allow_bond_change=bool(args.dissociation_allow_bond_change),
                burial_margin=float(args.burial_margin),
                work_dir=None,
                mace_model_path=str(args.mace_model_path),
                mace_device=str(args.mace_device),
                mace_dtype=str(args.mace_dtype),
                mace_head_name=infer_head_name(str(args.mace_model_path)),
            ),
            max_selected_primitives=int(args.max_selected_primitives),
            save_basin_ablation=True,
            basin_ablation_metrics=("rmsd", "mace_node_l2"),
            save_site_visualizations=False,
            save_raw_site_dictionary=False,
            save_selected_site_dictionary=False,
            primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.22),
        )
        t0 = time.perf_counter()
        result = run_adsorption_workflow(
            slab=slabs[slab_name],
            adsorbate=ads[ads_name],
            config=cfg,
            basin_relax_backend=relax_backend,
        )
        elapsed = time.perf_counter() - t0
        readiness = evaluate_adsorption_workflow_readiness(result)
        ab = json.loads((work_dir / "basin_ablation.json").read_text(encoding="utf-8"))
        n_rmsd = int(ab["metrics"]["rmsd"]["n_basins"])
        n_mace = int(ab["metrics"]["mace_node_l2"]["n_basins"])
        n_pose = int(result.summary["n_pose_frames"])
        case_score = (
            0.30 * float(1.0 if n_mace > 0 else 0.0)
            + 0.35 * ratio_score(n_basins=n_mace, n_pose=n_pose)
            + 0.25 * gap_score(n_mace=n_mace, n_rmsd=n_rmsd, n_pose=n_pose)
            + 0.05 * (float(readiness.score) / max(1.0, float(readiness.max_score)))
            + 0.05 * runtime_score(elapsed)
        )
        rows.append(
            {
                "slab": slab_name,
                "adsorbate": ads_name,
                "runtime_sec": elapsed,
                "n_pose_frames": n_pose,
                "n_basins_mace": n_mace,
                "n_basins_rmsd": n_rmsd,
                "paper_readiness_score": int(readiness.score),
                "paper_readiness_max_score": int(readiness.max_score),
                "case_score": float(case_score),
            }
        )

    final_score = 100.0 * sum(float(r["case_score"]) for r in rows) / float(max(1, len(rows)))
    payload = {
        "final_score": float(final_score),
        "n_cases": len(rows),
        "config": {
            "mace_model_path": str(args.mace_model_path),
            "mace_device": str(args.mace_device),
            "mace_dtype": str(args.mace_dtype),
            "max_selected_primitives": int(args.max_selected_primitives),
            "pose_n_rotations": int(args.pose_n_rotations),
            "pose_n_azimuth": int(args.pose_n_azimuth),
            "pose_n_shifts": int(args.pose_n_shifts),
            "pose_max_per_site": int(args.pose_max_per_site),
            "relax_maxf": float(args.relax_maxf),
            "relax_steps": int(args.relax_steps),
            "energy_window_ev": float(args.energy_window_ev),
            "mace_node_l2_threshold": float(args.mace_node_l2_threshold),
            "desorption_min_bonds": int(args.desorption_min_bonds),
            "surface_reconstruction_max_disp": float(args.surface_reconstruction_max_disp),
            "dissociation_allow_bond_change": bool(args.dissociation_allow_bond_change),
            "burial_margin": float(args.burial_margin),
        },
        "rows": rows,
    }
    (out_root / "benchmark_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINAL_SCORE: {final_score:.6f}")
    print(json.dumps({"final_score": final_score, "n_cases": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
