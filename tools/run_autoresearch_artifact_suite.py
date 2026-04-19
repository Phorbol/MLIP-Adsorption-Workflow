from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
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


DEFAULT_MACE_MODEL_CANDIDATES = (
    "/root/.cache/mace/mace-omat-0-small.model",
    "/root/.cache/mace/mace-mh-1.model",
    "/root/.cache/mace/MACE-OFF23_small.model",
    "/root/.cache/mace/MACE-OFF24_medium.model",
)


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


def build_config(
    work_dir: Path,
    *,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
    max_selected_primitives: int,
) -> AdsorptionWorkflowConfig:
    return AdsorptionWorkflowConfig(
        work_dir=work_dir,
        surface_preprocessor=make_default_surface_preprocessor(),
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
            relax_steps=80,
            energy_window_ev=2.0,
            dedup_metric="mace_node_l2",
            mace_node_l2_threshold=2.0,
            desorption_min_bonds=0,
            work_dir=None,
            mace_model_path=mace_model_path,
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
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


def infer_mace_head_name(model_path: str | None) -> str | None:
    if not model_path:
        return None
    name = Path(model_path).name.lower()
    if "omat" in name:
        return "omat_pbe"
    if "omol" in name:
        return "omol"
    return None


def run_cases(
    *,
    out_root: Path,
    cases: list[tuple[str, str]],
    slabs: dict,
    adsorbates: dict,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
    max_selected_primitives: int,
) -> list[dict]:
    rows = []
    head_name = infer_mace_head_name(mace_model_path)
    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=mace_model_path,
            device=str(mace_device),
            dtype=str(mace_dtype),
            max_edges_per_batch=15000,
            head_name=head_name,
            strict=True,
        )
    )
    for slab_name, ads_name in cases:
        case_dir = out_root / slab_name / ads_name
        cfg = build_config(
            case_dir,
            mace_model_path=mace_model_path,
            mace_device=mace_device,
            mace_dtype=mace_dtype,
            max_selected_primitives=max_selected_primitives,
        )
        result = run_adsorption_workflow(
            slab=slabs[slab_name],
            adsorbate=adsorbates[ads_name],
            config=cfg,
            basin_relax_backend=relax_backend,
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


def resolve_mace_model_path(cli_value: str) -> tuple[str | None, str]:
    val = str(cli_value).strip()
    if val:
        p = Path(val)
        return (str(p) if p.exists() else None), "cli"
    env_val = str(os.environ.get("AE_MACE_MODEL_PATH", "")).strip()
    if env_val:
        p = Path(env_val)
        return (str(p) if p.exists() else None), "env"
    for cand in DEFAULT_MACE_MODEL_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return str(p), "auto"
    return None, "none"


def runtime_manifest(
    *,
    mace_model_path: str | None,
    model_source: str,
    mace_device: str,
    mace_device_effective: str,
    mace_dtype: str,
    out_root: Path,
) -> dict:
    py = shutil.which("python")
    payload = {
        "cwd": Path.cwd().as_posix(),
        "python_executable": py,
        "mace_model_path": mace_model_path,
        "mace_model_source": model_source,
        "mace_device_requested": str(mace_device),
        "mace_device_effective": str(mace_device_effective),
        "mace_dtype": str(mace_dtype),
        "ae_disable_mace": str(os.environ.get("AE_DISABLE_MACE", "")),
    }
    try:
        import torch  # type: ignore

        payload["torch_version"] = str(torch.__version__)
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        payload["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import mace  # type: ignore

        payload["mace_version"] = str(getattr(mace, "__version__", "unknown"))
    except Exception as exc:
        payload["mace_import_error"] = f"{type(exc).__name__}: {exc}"
    out_path = out_root / "runtime_manifest.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--max-selected-primitives", type=int, default=24)
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

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

    mace_model_path, model_source = resolve_mace_model_path(args.mace_model_path)
    runtime_info = runtime_manifest(
        mace_model_path=mace_model_path,
        model_source=model_source,
        mace_device=device_requested,
        mace_device_effective=device_effective,
        mace_dtype=str(args.mace_dtype),
        out_root=out_root,
    )

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
        mace_device=device_effective,
        mace_dtype=str(args.mace_dtype),
        max_selected_primitives=int(args.max_selected_primitives),
    )
    real_rows = run_cases(
        out_root=real_dir,
        cases=real_cases(),
        slabs=slabs,
        adsorbates=adsorbates,
        mace_model_path=mace_model_path,
        mace_device=device_effective,
        mace_dtype=str(args.mace_dtype),
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
        "mace_model_source": model_source,
        "mace_device_requested": device_requested,
        "mace_device_effective": device_effective,
        "mace_dtype": str(args.mace_dtype),
        "runtime_manifest_json": (out_root / "runtime_manifest.json").as_posix(),
        "torch_cuda_available": runtime_info.get("torch_cuda_available"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
