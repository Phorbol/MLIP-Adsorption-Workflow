from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ase.build import bcc110, bulk, fcc100, fcc111, fcc211, molecule, surface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, evaluate_adsorption_workflow_readiness, run_adsorption_workflow
from tests.chemistry_cases import get_test_adsorbate_cases


def build_slab_cases():
    return {
        "fcc111": fcc111("Pt", size=(3, 3, 3), vacuum=10.0),
        "fcc100": fcc100("Pt", size=(3, 3, 3), vacuum=10.0),
        "fcc211": fcc211("Pt", size=(6, 3, 3), vacuum=10.0),
        "bcc110": bcc110("Fe", size=(3, 3, 3), vacuum=10.0),
        "cu321": surface(bulk("Cu", "fcc", a=3.6, cubic=True), (3, 2, 1), layers=3, vacuum=10.0).repeat((2, 1, 1)),
    }


def default_cases():
    return [
        ("fcc111", "CO"),
        ("fcc100", "H2O"),
        ("fcc211", "CH3OH"),
        ("bcc110", "C2H4"),
        ("cu321", "C6H6"),
        ("fcc111", "glucose_chain_like"),
        ("fcc100", "glucose_ring_like"),
        ("fcc211", "glycine_like"),
        ("bcc110", "dipeptide_like"),
        ("cu321", "p_nitrochlorobenzene_like"),
        ("fcc111", "p_nitrobenzoic_acid_like"),
    ]


def build_config(work_dir: Path) -> AdsorptionWorkflowConfig:
    return AdsorptionWorkflowConfig(
        work_dir=work_dir,
        surface_preprocessor=SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=0.6),
            fallback_detector=VoxelFloodDetector(spacing=0.8),
            target_surface_fraction=None,
            target_count_mode="off",
        ),
        pose_sampler_config=PoseSamplerConfig(
            n_rotations=4,
            n_azimuth=8,
            n_shifts=2,
            shift_radius=0.2,
            min_height=1.5,
            max_height=3.0,
            height_step=0.2,
            max_poses_per_site=4,
            random_seed=0,
        ),
        basin_config=BasinConfig(
            relax_maxf=0.1,
            relax_steps=1,
            energy_window_ev=2.0,
            desorption_min_bonds=0,
            work_dir=None,
        ),
        max_primitives=None,
        max_selected_primitives=24,
        save_basin_dictionary=True,
        save_basin_ablation=True,
        basin_ablation_metrics=("signature_only", "rmsd"),
        save_site_visualizations=True,
        save_raw_site_dictionary=True,
        save_selected_site_dictionary=True,
        primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.22),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch")
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    slabs = build_slab_cases()
    adsorbates = get_test_adsorbate_cases()
    rows = []
    for slab_name, ads_name in default_cases():
        case_dir = out_root / "workflow_matrix" / slab_name / ads_name
        cfg = build_config(case_dir)
        result = run_adsorption_workflow(
            slab=slabs[slab_name],
            adsorbate=adsorbates[ads_name],
            config=cfg,
        )
        readiness = evaluate_adsorption_workflow_readiness(result)
        row = {
            "slab": slab_name,
            "adsorbate": ads_name,
            "n_primitives": int(result.summary["n_primitives"]),
            "n_pose_frames": int(result.summary["n_pose_frames"]),
            "n_basins": int(result.summary["n_basins"]),
            "n_nodes": int(result.summary["n_nodes"]),
            "paper_readiness_score": int(readiness.score),
            "paper_readiness_max_score": int(readiness.max_score),
            "work_dir": case_dir.as_posix(),
        }
        rows.append(row)

    summary_json = out_root / "workflow_matrix_summary.json"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_csv = out_root / "workflow_matrix_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary_json": summary_json.as_posix(), "summary_csv": summary_csv.as_posix(), "n_cases": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
