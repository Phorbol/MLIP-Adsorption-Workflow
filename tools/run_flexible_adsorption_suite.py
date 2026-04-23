from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.conformer_md import ConformerMDSamplerConfig, resolve_selection_profile
from adsorption_ensemble.relax import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    DEFAULT_MACE_HEAD_NAME,
    evaluate_adsorption_workflow_readiness,
    make_adsorption_workflow_config,
    make_default_surface_preprocessor,
    make_sampling_schedule,
    plan_flex_sampling_budget,
    run_adsorption_workflow,
)
from tools.run_autoresearch_artifact_suite import resolve_mace_model_path, runtime_manifest
from tools.run_comprehensive_adsorption_matrix import build_adsorbates, build_slabs
from tools.run_production_case_suite import load_cnb_isomer_adsorbates


@dataclass(frozen=True)
class FlexibleCaseSpec:
    case_id: str
    slab_key: str
    adsorbate_key: str
    rationale: str
    source_note: str
    generator_backend: str = "xtb_md"


def default_cases() -> tuple[FlexibleCaseSpec, ...]:
    return (
        FlexibleCaseSpec(
            case_id="TiO2_110__CH3COOH",
            slab_key="TiO2_110",
            adsorbate_key="g2_CH3COOH",
            rationale="O-rich acid on oxide; stresses multi-anchor adsorption, orientation competition, and collision handling.",
            source_note="DockOnSurf-inspired oxide + multifunctional polar adsorbate class.",
            generator_backend="rdkit_embed",
        ),
        FlexibleCaseSpec(
            case_id="CuNi_fcc111_alloy__CH3CONH2",
            slab_key="CuNi_fcc111_alloy",
            adsorbate_key="g2_CH3CONH2",
            rationale="Flexible amide on heterogeneous alloy; stresses conformer/orientation/site coupling on mixed local environments.",
            source_note="DockOnSurf-inspired conformer × site × orientation decomposition on chemically heterogeneous surfaces.",
            generator_backend="rdkit_embed",
        ),
        FlexibleCaseSpec(
            case_id="Pt_fcc111__CH3CH2OH",
            slab_key="Pt_fcc111",
            adsorbate_key="g2_CH3CH2OH",
            rationale="Small rotatable alcohol on close-packed metal; tests that the flexible workflow still behaves sensibly on a lower-complexity case.",
            source_note="Practical small-flexible benchmark for production smoke-to-real validation.",
            generator_backend="rdkit_embed",
        ),
    )


def _parse_case_ids(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _resolve_effective_device(requested: str, *, require_cuda: bool) -> str:
    device_req = str(requested).strip() or "cpu"
    if not device_req.lower().startswith("cuda"):
        return device_req
    try:
        import torch  # type: ignore
    except Exception as exc:
        if bool(require_cuda):
            raise RuntimeError("CUDA was requested but torch import failed.") from exc
        return "cpu"
    if bool(torch.cuda.is_available()):
        return device_req
    if bool(require_cuda):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return "cpu"


def _cap_budget_for_runtime(budget, *, device_effective: str):
    if str(device_effective).lower().startswith("cuda"):
        return budget
    return replace(
        budget,
        md_time_ps=min(float(budget.md_time_ps), 8.0),
        md_runs=min(int(budget.md_runs), 2),
        preselect_k=min(int(budget.preselect_k), 48),
        target_final_k=min(int(budget.target_final_k), 4),
        fps_rounds=min(int(budget.fps_rounds), 4),
        fps_round_size=min(int(budget.fps_round_size), 16),
    )


def _build_adsorbates() -> dict[str, object]:
    adsorbates = dict(build_adsorbates())
    adsorbates.update(load_cnb_isomer_adsorbates("examples"))
    return adsorbates


def _estimate_surface_complexity(slab) -> tuple[int, int | None]:
    pre = make_default_surface_preprocessor()
    ctx = pre.build_context(slab.copy())
    n_surface_atoms = int(len(ctx.detection.surface_atom_ids))
    return n_surface_atoms, None


def _build_conformer_config(
    *,
    case_dir: Path,
    budget,
    generator_backend: str,
    mace_model_path: str | None,
    mace_device_requested: str,
    mace_dtype: str,
    mace_head_name: str,
) -> ConformerMDSamplerConfig:
    cfg = ConformerMDSamplerConfig()
    cfg.generator.backend = str(generator_backend)
    cfg.md.temperature_k = 450.0
    cfg.md.time_ps = float(budget.md_time_ps)
    cfg.md.step_fs = 1.0
    cfg.md.dump_fs = 50.0
    cfg.md.n_runs = int(budget.md_runs)
    cfg.md.seed_mode = "increment_per_run"
    cfg.generator.rdkit.num_confs = int(max(32, min(128, 6 * int(budget.target_final_k))))
    cfg.generator.rdkit.prune_rms_thresh = 0.15
    cfg.generator.rdkit.optimize_forcefield = "mmff"
    cfg.descriptor.backend = "mace"
    cfg.descriptor.mace.model_path = mace_model_path
    cfg.descriptor.mace.device = str(mace_device_requested)
    cfg.descriptor.mace.dtype = "float64"
    cfg.descriptor.mace.head_name = str(mace_head_name)
    cfg.descriptor.mace.enable_cueq = False
    cfg.relax.backend = "mace_energy"
    cfg.relax.mace.model_path = mace_model_path
    cfg.relax.mace.device = str(mace_device_requested)
    cfg.relax.mace.dtype = str(mace_dtype)
    cfg.relax.mace.head_name = str(mace_head_name)
    cfg.relax.mace.enable_cueq = bool(str(mace_device_requested).lower().startswith("cuda"))
    cfg.selection.preselect_k = int(budget.preselect_k)
    cfg.selection.pair_energy_gap_ev = 0.01
    cfg.output.save_all_frames = True
    cfg.output.work_dir = case_dir / "conformer_md"
    cfg = resolve_selection_profile(
        cfg,
        profile=str(budget.selection_profile),
        target_final_k=int(budget.target_final_k),
    )
    return cfg


def _build_workflow_config(
    *,
    case_dir: Path,
    adsorbate,
    budget,
    mace_model_path: str | None,
    mace_device_effective: str,
    mace_dtype: str,
    mace_head_name: str,
    generator_backend: str,
):
    schedule = make_sampling_schedule("multistage_default")
    schedule.pre_relax_selection.max_candidates = 64
    cfg = make_adsorption_workflow_config(
        work_dir=case_dir,
        placement_mode="anchor_free",
        single_atom=(len(adsorbate) == 1),
        exhaustive_pose_sampling=True,
        dedup_metric="mace_node_l2",
        signature_mode="provenance",
        pose_overrides={
            "n_rotations": 8,
            "n_azimuth": 12,
            "n_shifts": 3,
            "shift_radius": 0.25,
            "min_height": 1.2,
            "max_height": 3.8,
            "height_step": 0.10,
            "max_poses_per_site": 6,
            "adaptive_height_fallback": True,
            "adaptive_height_fallback_step": 0.20,
            "adaptive_height_fallback_max_extra": 1.60,
            "adaptive_height_fallback_contact_slack": 0.60,
        },
        basin_overrides={
            "dedup_cluster_method": "greedy",
            "mace_model_path": mace_model_path,
            "mace_device": str(mace_device_effective),
            "mace_dtype": "float64",
            "mace_enable_cueq": False,
            "mace_head_name": str(mace_head_name),
            "desorption_min_bonds": 1,
            "surface_reconstruction_enabled": False,
            "energy_window_ev": 2.5,
        },
    )
    cfg.surface_preprocessor = make_default_surface_preprocessor(target_count_mode="adaptive", target_surface_fraction=0.25)
    cfg.pre_relax_selection = schedule.pre_relax_selection
    cfg.basin_config.post_relax_selection = schedule.post_relax_selection
    cfg.max_selected_primitives = 36
    cfg.run_conformer_search = True
    cfg.conformer_config = _build_conformer_config(
        case_dir=case_dir,
        budget=budget,
        generator_backend=generator_backend,
        mace_model_path=mace_model_path,
        mace_device_requested="cuda" if str(mace_device_effective).lower().startswith("cpu") else str(mace_device_effective),
        mace_dtype=mace_dtype,
        mace_head_name=mace_head_name,
    )
    cfg.conformer_job_name = "conformer_search"
    return cfg


def _write_rows(rows: list[dict], out_root: Path) -> None:
    summary_json = out_root / "flexible_adsorption_suite_summary.json"
    summary_csv = out_root / "flexible_adsorption_suite_summary.csv"
    summary_md = out_root / "flexible_adsorption_suite_summary.md"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Flexible Adsorption Suite",
        "",
        "| case | generator_backend | mace_device_requested | mace_device_effective | n_conformers | n_surface_atoms | n_raw_primitives | n_selected_primitives | n_basis_primitives | n_pose_frames | n_pose_frames_selected_for_basin | n_basins | n_nodes | readiness | work_dir |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["generator_backend"]),
                    str(row["mace_device_requested"]),
                    str(row["mace_device_effective"]),
                    str(row["n_conformers"]),
                    str(row["n_surface_atoms"]),
                    str(row["n_raw_primitives"]),
                    str(row["n_selected_primitives"]),
                    str(row["n_basis_primitives"]),
                    str(row["n_pose_frames"]),
                    str(row["n_pose_frames_selected_for_basin"]),
                    str(row["n_basins"]),
                    str(row["n_nodes"]),
                    f'{row["paper_readiness_score"]}/{row["paper_readiness_max_score"]}',
                    str(row["work_dir"]),
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/flexible_adsorption_suite")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-mh-1.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--mace-head-name", type=str, default=DEFAULT_MACE_HEAD_NAME)
    parser.add_argument("--require-cuda", action="store_true", default=False)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path, model_source = resolve_mace_model_path(args.mace_model_path)
    device_requested = str(args.mace_device)
    device_effective = _resolve_effective_device(device_requested, require_cuda=bool(args.require_cuda))
    runtime_manifest(
        mace_model_path=model_path,
        model_source=model_source,
        mace_device=device_requested,
        mace_device_effective=device_effective,
        mace_dtype=str(args.mace_dtype),
        out_root=out_root,
    )

    slabs = build_slabs()
    adsorbates = _build_adsorbates()
    case_specs = list(default_cases())
    if str(args.cases).strip():
        requested = set(_parse_case_ids(args.cases))
        case_specs = [spec for spec in case_specs if spec.case_id in requested]
    if not case_specs:
        raise ValueError("No flexible adsorption cases selected.")

    strict_relax = bool(str(device_effective).lower().startswith("cuda"))
    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=str(device_effective),
            dtype=str(args.mace_dtype),
            head_name=str(args.mace_head_name),
            enable_cueq=bool(str(device_effective).lower().startswith("cuda")),
            strict=bool(strict_relax),
            max_edges_per_batch=100000,
        )
    )

    rows: list[dict] = []
    for spec in case_specs:
        slab = slabs[spec.slab_key].copy()
        adsorbate = adsorbates[spec.adsorbate_key].copy()
        n_surface_atoms, n_site_primitives = _estimate_surface_complexity(slab)
        base_budget = plan_flex_sampling_budget(
            adsorbate,
            n_surface_atoms=n_surface_atoms,
            n_site_primitives=n_site_primitives,
        )
        budget = _cap_budget_for_runtime(base_budget, device_effective=device_effective)
        case_dir = out_root / spec.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case_spec.json").write_text(
            json.dumps(
                {
                    **asdict(spec),
                    "base_budget": asdict(base_budget),
                    "resolved_budget": asdict(budget),
                    "mace_device_requested": device_requested,
                    "mace_device_effective": device_effective,
                    "mace_head_name": str(args.mace_head_name),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        cfg = _build_workflow_config(
            case_dir=case_dir,
            adsorbate=adsorbate,
            budget=budget,
            mace_model_path=model_path,
            mace_device_effective=device_effective,
            mace_dtype=str(args.mace_dtype),
            mace_head_name=str(args.mace_head_name),
            generator_backend=str(spec.generator_backend),
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
                "case": spec.case_id,
                "generator_backend": spec.generator_backend,
                "mace_device_requested": device_requested,
                "mace_device_effective": device_effective,
                "mace_head_name": str(args.mace_head_name),
                "rationale": spec.rationale,
                "source_note": spec.source_note,
                "n_conformers": int(result.summary.get("n_conformers", 1)),
                "n_surface_atoms": int(result.summary["n_surface_atoms"]),
                "n_raw_primitives": int(result.summary["n_raw_primitives"]),
                "n_selected_primitives": int(result.summary["n_primitives"]),
                "n_basis_primitives": int(result.summary["n_basis_primitives"]),
                "n_pose_frames": int(result.summary["n_pose_frames"]),
                "n_pose_frames_selected_for_basin": int(
                    result.summary.get("n_pose_frames_selected_for_basin", result.summary["n_pose_frames"])
                ),
                "n_basins": int(result.summary["n_basins"]),
                "n_nodes": int(result.summary["n_nodes"]),
                "paper_readiness_score": int(readiness.score),
                "paper_readiness_max_score": int(readiness.max_score),
                "work_dir": case_dir.as_posix(),
            }
        )
    _write_rows(rows, out_root)
    print((out_root / "flexible_adsorption_suite_summary.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
