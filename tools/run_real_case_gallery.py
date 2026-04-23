from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ase.collections import g2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.conformer_md import ConformerMDSamplerConfig, resolve_selection_profile
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    evaluate_adsorption_workflow_readiness,
    make_adsorption_workflow_config,
    make_sampling_schedule,
    plan_flex_sampling_budget,
    run_adsorption_workflow,
)
from tools.run_autoresearch_artifact_suite import resolve_mace_model_path, runtime_manifest
from tools.run_comprehensive_adsorption_matrix import build_adsorbates, build_slabs


DEFAULT_CASES: tuple[tuple[str, str], ...] = (
    ("TiO2_110", "CH3CONH2"),
    ("CuNi_fcc111_alloy", "CH3COOH"),
    ("Pt_fcc211", "p_nitrobenzoic_acid_like"),
    ("Pt_fcc111", "dipeptide_like"),
)

EXTRA_G2_ADSORBATES: tuple[str, ...] = (
    "CH3COOH",
    "CH3CONH2",
    "CH3CH2OH",
    "HCOOCH3",
)


def _parse_cases(raw: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for token in str(raw).split(","):
        piece = token.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Invalid case spec: {piece}. Expected slab:adsorbate")
        slab_name, ads_name = piece.split(":", 1)
        items.append((slab_name.strip(), ads_name.strip()))
    return items


def _head_name(model_path: str | None) -> str | None:
    name = "" if model_path is None else Path(model_path).name.lower()
    if "omat" in name:
        return "omat_pbe"
    if "omol" in name:
        return "omol"
    return None


def _build_gallery_adsorbates() -> dict[str, object]:
    adsorbates = dict(build_adsorbates())
    for name in EXTRA_G2_ADSORBATES:
        try:
            adsorbates[name] = g2[name].copy()
        except Exception:
            continue
    return adsorbates


def _configure_flexible_search(
    cfg,
    *,
    case_dir: Path,
    budget,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
    head_name: str | None,
) -> None:
    conformer_cfg = ConformerMDSamplerConfig()
    conformer_cfg.md.temperature_k = 450.0
    conformer_cfg.md.time_ps = float(budget.md_time_ps)
    conformer_cfg.md.step_fs = 1.0
    conformer_cfg.md.dump_fs = 50.0
    conformer_cfg.md.n_runs = int(budget.md_runs)
    conformer_cfg.md.seed_mode = "increment_per_run"
    conformer_cfg.descriptor.backend = "mace"
    conformer_cfg.descriptor.mace.model_path = mace_model_path
    conformer_cfg.descriptor.mace.device = str(mace_device)
    conformer_cfg.descriptor.mace.dtype = str(mace_dtype)
    conformer_cfg.descriptor.mace.head_name = str(head_name or "Default")
    conformer_cfg.descriptor.mace.enable_cueq = True
    conformer_cfg.relax.backend = "mace_energy"
    conformer_cfg.relax.mace.model_path = mace_model_path
    conformer_cfg.relax.mace.device = str(mace_device)
    conformer_cfg.relax.mace.dtype = str(mace_dtype)
    conformer_cfg.relax.mace.head_name = str(head_name or "Default")
    conformer_cfg.relax.mace.enable_cueq = True
    conformer_cfg.selection.preselect_k = int(budget.preselect_k)
    conformer_cfg.selection.energy_window_ev = 0.30
    conformer_cfg.selection.rmsd_threshold = 0.08
    conformer_cfg.selection.fps_convergence_enable = True
    conformer_cfg.selection.fps_convergence_grid_bins = 16
    conformer_cfg.selection.fps_convergence_min_rounds = int(budget.fps_rounds)
    conformer_cfg.selection.fps_round_size = int(budget.fps_round_size)
    conformer_cfg.selection.fps_convergence_patience = 3
    conformer_cfg.output.save_all_frames = True
    conformer_cfg.output.work_dir = case_dir / "conformer_md"
    conformer_cfg = resolve_selection_profile(
        conformer_cfg,
        profile=str(budget.selection_profile),
        target_final_k=int(budget.target_final_k),
    )
    cfg.run_conformer_search = True
    cfg.conformer_config = conformer_cfg
    cfg.conformer_job_name = "conformer_search"


def _build_case_config(
    *,
    case_dir: Path,
    adsorbate,
    budget,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
) :
    schedule = make_sampling_schedule("multistage_default")
    size_based_pose_overrides = {}
    if len(adsorbate) >= 12:
        size_based_pose_overrides = {
            "n_rotations": 6,
            "n_azimuth": 12,
            "n_shifts": 3,
            "shift_radius": 0.25,
            "min_height": 1.4,
            "max_height": 3.6,
            "height_step": 0.15,
            "max_poses_per_site": 6,
        }
    cfg = make_adsorption_workflow_config(
        work_dir=case_dir,
        placement_mode="anchor_free",
        single_atom=(len(adsorbate) == 1),
        exhaustive_pose_sampling=bool(schedule.exhaustive_pose_sampling),
        dedup_metric="binding_surface_distance",
        signature_mode="provenance",
        pose_overrides=size_based_pose_overrides,
        basin_overrides={
            "mace_model_path": mace_model_path,
            "mace_device": str(mace_device),
            "mace_dtype": str(mace_dtype),
            "mace_head_name": str(_head_name(mace_model_path) or "Default"),
            "relax_maxf": 0.10,
            "relax_steps": 80,
            "energy_window_ev": 2.5,
            "desorption_min_bonds": 1,
        },
    )
    cfg.pre_relax_selection = schedule.pre_relax_selection
    cfg.basin_config.post_relax_selection = schedule.post_relax_selection
    if len(adsorbate) >= 12:
        cfg.max_selected_primitives = 36
    if bool(budget.run_conformer_search):
        _configure_flexible_search(
            cfg,
            case_dir=case_dir,
            budget=budget,
            mace_model_path=mace_model_path,
            mace_device=mace_device,
            mace_dtype=mace_dtype,
            head_name=_head_name(mace_model_path),
        )
    return cfg


def _write_rows(rows: list[dict], out_root: Path) -> None:
    (out_root / "real_case_gallery_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        with (out_root / "real_case_gallery_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Real Case Gallery",
        "",
        "| case | run_conformer_search | n_conformers | n_surface_atoms | n_raw_primitives | n_selected_primitives | n_basis_primitives | n_pose_frames | n_pose_frames_selected_for_basin | n_basins | n_nodes | readiness | work_dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(int(bool(row["run_conformer_search"]))),
                    str(row["n_conformers"]),
                    str(row["n_surface_atoms"]),
                    str(row["n_raw_primitives"]),
                    str(row["n_selected_primitives"]),
                    str(row["n_basis_primitives"]),
                    str(row["n_pose_frames"]),
                    str(row["n_pose_frames_selected_for_basin"]),
                    str(row["n_basins"]),
                    str(row["n_nodes"]),
                    f"{row['paper_readiness_score']}/{row['paper_readiness_max_score']}",
                    str(row["work_dir"]),
                ]
            )
            + " |"
        )
    (out_root / "real_case_gallery_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/real_case_gallery_20260406")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path, model_source = resolve_mace_model_path(args.mace_model_path)
    device_requested = str(args.mace_device)
    device_effective = str(device_requested)
    if str(device_requested).lower().startswith("cuda"):
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("CUDA is required for this gallery run, but torch import failed.") from exc
        if not bool(torch.cuda.is_available()) and bool(args.require_cuda):
            raise RuntimeError("CUDA is required for this gallery run, but torch.cuda.is_available() is False.")

    runtime_manifest(
        mace_model_path=model_path,
        model_source=model_source,
        mace_device=device_requested,
        mace_device_effective=device_effective,
        mace_dtype=str(args.mace_dtype),
        out_root=out_root,
    )

    slabs = build_slabs()
    adsorbates = _build_gallery_adsorbates()
    cases = _parse_cases(args.cases) if str(args.cases).strip() else list(DEFAULT_CASES)
    missing = [(slab_name, ads_name) for slab_name, ads_name in cases if slab_name not in slabs or ads_name not in adsorbates]
    if missing:
        raise KeyError(f"Unknown cases requested: {missing}")

    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=device_effective,
            dtype=str(args.mace_dtype),
            max_edges_per_batch=20000,
            head_name=_head_name(model_path),
            enable_cueq=True,
            strict=True,
        )
    )

    rows: list[dict] = []
    for slab_name, ads_name in cases:
        slab = slabs[slab_name]
        adsorbate = adsorbates[ads_name].copy()
        case_dir = out_root / slab_name / ads_name
        budget = plan_flex_sampling_budget(
            adsorbate,
            n_surface_atoms=len(slab),
            n_site_primitives=36 if len(adsorbate) >= 12 else 24,
        )
        cfg = _build_case_config(
            case_dir=case_dir,
            adsorbate=adsorbate,
            budget=budget,
            mace_model_path=model_path,
            mace_device=device_effective,
            mace_dtype=str(args.mace_dtype),
        )
        result = run_adsorption_workflow(
            slab=slab,
            adsorbate=adsorbate,
            config=cfg,
            basin_relax_backend=relax_backend,
        )
        readiness = evaluate_adsorption_workflow_readiness(result)
        rows.append(
            {
                "case": f"{slab_name}__{ads_name}",
                "slab": slab_name,
                "adsorbate": ads_name,
                "run_conformer_search": bool(result.summary.get("run_conformer_search", False)),
                "flex_score": float(budget.score),
                "n_conformers": int(result.summary.get("n_conformers", 1)),
                "n_surface_atoms": int(result.summary["n_surface_atoms"]),
                "n_raw_primitives": int(result.summary["n_raw_primitives"]),
                "n_selected_primitives": int(result.summary["n_primitives"]),
                "n_basis_primitives": int(result.summary["n_basis_primitives"]),
                "n_pose_frames": int(result.summary["n_pose_frames"]),
                "n_pose_frames_selected_for_basin": int(result.summary.get("n_pose_frames_selected_for_basin", result.summary["n_pose_frames"])),
                "n_basins": int(result.summary["n_basins"]),
                "n_nodes": int(result.summary["n_nodes"]),
                "paper_readiness_score": int(readiness.score),
                "paper_readiness_max_score": int(readiness.max_score),
                "surface_classification": json.dumps(result.summary.get("surface_classification", {}), ensure_ascii=False),
                "surface_diagnostics": json.dumps(result.summary.get("surface_diagnostics", {}), ensure_ascii=False),
                "work_dir": case_dir.as_posix(),
            }
        )

    _write_rows(rows, out_root)
    print(json.dumps({"out_root": out_root.as_posix(), "n_cases": len(rows), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
