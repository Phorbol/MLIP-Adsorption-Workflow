from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

from ase.build import bcc110, bulk, fcc100, fcc111, fcc211, surface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, evaluate_adsorption_workflow_readiness, run_adsorption_workflow
from tests.chemistry_cases import get_test_adsorbate_cases


def build_slab_cases():
    return {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=10.0),
        "bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=10.0),
        "cu321": surface(bulk("Cu", "fcc", a=3.6, cubic=True), (3, 2, 1), layers=4, vacuum=10.0).repeat((2, 2, 1)),
    }


def workflow_matrix_cases():
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


def real_cases():
    return [
        ("fcc111", "CO"),
        ("fcc111", "H2O"),
        ("fcc111", "NH3"),
        ("fcc111", "C2H2"),
        ("fcc111", "C2H4"),
        ("fcc111", "C2H6"),
        ("fcc111", "CH3OH"),
        ("fcc111", "C6H6"),
        ("fcc111", "glucose_chain_like"),
        ("fcc100", "H2O"),
        ("fcc100", "C2H4"),
        ("fcc100", "C2H6"),
        ("fcc100", "glucose_ring_like"),
        ("fcc211", "CH3OH"),
        ("fcc211", "glycine_like"),
        ("fcc211", "dipeptide_like"),
        ("bcc110", "C2H4"),
        ("bcc110", "C2H2"),
        ("bcc110", "dipeptide_like"),
        ("cu321", "C6H6"),
        ("cu321", "p_nitrochlorobenzene_like"),
        ("cu321", "p_nitrobenzoic_acid_like"),
    ]


def build_config(work_dir: Path, *, mace_model_path: str | None, max_selected_primitives: int) -> AdsorptionWorkflowConfig:
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
            mace_model_path=mace_model_path,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=15000,
            mace_layers_to_keep=-1,
        ),
        max_primitives=None,
        max_selected_primitives=max_selected_primitives,
        save_basin_dictionary=True,
        save_basin_ablation=True,
        basin_ablation_metrics=("signature_only", "rmsd", "mace_node_l2"),
        save_site_visualizations=True,
        save_raw_site_dictionary=True,
        save_selected_site_dictionary=True,
        primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.22),
    )


def run_cases(
    *,
    out_root: Path,
    cases: list[tuple[str, str]],
    slabs: dict,
    adsorbates: dict,
    mace_model_path: str | None,
    max_selected_primitives: int,
) -> list[dict]:
    rows = []
    for slab_name, ads_name in cases:
        case_dir = out_root / slab_name / ads_name
        cfg = build_config(case_dir, mace_model_path=mace_model_path, max_selected_primitives=max_selected_primitives)
        result = run_adsorption_workflow(
            slab=slabs[slab_name],
            adsorbate=adsorbates[ads_name],
            config=cfg,
        )
        readiness = evaluate_adsorption_workflow_readiness(result)
        row = {
            "slab": slab_name,
            "adsorbate": ads_name,
            "n_surface_atoms": int(result.summary["n_surface_atoms"]),
            "n_raw_primitives": int(result.summary["n_raw_primitives"]),
            "n_selected_primitives": int(result.summary["n_primitives"]),
            "n_basis_primitives": int(result.summary["n_basis_primitives"]),
            "n_pose_frames": int(result.summary["n_pose_frames"]),
            "n_basins": int(result.summary["n_basins"]),
            "n_nodes": int(result.summary["n_nodes"]),
            "paper_readiness_score": int(readiness.score),
            "paper_readiness_max_score": int(readiness.max_score),
            "work_dir": case_dir.as_posix(),
        }
        rows.append(row)
    return rows


def write_rows(rows: list[dict], *, out_json: Path, out_csv: Path) -> None:
    if not rows:
        out_json.write_text("[]\n", encoding="utf-8")
        out_csv.write_text("", encoding="utf-8")
        return
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--max-selected-primitives", type=int, default=24)
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mace_model_path = str(args.mace_model_path).strip()
    if not mace_model_path:
        mace_model_path = str(os.environ.get("AE_MACE_MODEL_PATH", "")).strip()
    mace_model_path = mace_model_path if mace_model_path else None

    slabs = build_slab_cases()
    adsorbates = get_test_adsorbate_cases()

    matrix_dir = out_root / "workflow_matrix"
    real_dir = out_root / "real_cases"
    matrix_rows = run_cases(
        out_root=matrix_dir,
        cases=workflow_matrix_cases(),
        slabs=slabs,
        adsorbates=adsorbates,
        mace_model_path=mace_model_path,
        max_selected_primitives=int(args.max_selected_primitives),
    )
    real_rows = run_cases(
        out_root=real_dir,
        cases=real_cases(),
        slabs=slabs,
        adsorbates=adsorbates,
        mace_model_path=mace_model_path,
        max_selected_primitives=int(args.max_selected_primitives),
    )

    matrix_json = out_root / "workflow_matrix_summary.json"
    matrix_csv = out_root / "workflow_matrix_summary.csv"
    real_json = out_root / "real_cases_summary.json"
    real_csv = out_root / "real_cases_summary.csv"
    write_rows(matrix_rows, out_json=matrix_json, out_csv=matrix_csv)
    write_rows(real_rows, out_json=real_json, out_csv=real_csv)

    payload = {
        "workflow_matrix_summary_json": matrix_json.as_posix(),
        "workflow_matrix_summary_csv": matrix_csv.as_posix(),
        "real_cases_summary_json": real_json.as_posix(),
        "real_cases_summary_csv": real_csv.as_posix(),
        "n_workflow_matrix_cases": len(matrix_rows),
        "n_real_cases": len(real_rows),
        "mace_model_path": mace_model_path,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
