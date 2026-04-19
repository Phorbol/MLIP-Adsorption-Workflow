from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import fcc111, fcc211, molecule
from ase.io import read

from adsorption_ensemble.relax import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    DEFAULT_MACE_HEAD_NAME,
    evaluate_adsorption_workflow_readiness,
    generate_adsorption_ensemble,
    make_adsorption_workflow_config,
    make_default_surface_preprocessor,
    make_sampling_schedule,
    run_adsorption_workflow,
)
from tests.chemistry_cases import get_test_adsorbate_cases
from tools.run_autoresearch_artifact_suite import resolve_mace_model_path, runtime_manifest


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    slab_key: str
    adsorbate_key: str
    mode: str
    notes: str = ""


def _center_atoms(atoms: Atoms) -> Atoms:
    out = atoms.copy()
    pos = np.asarray(out.get_positions(), dtype=float)
    pos -= np.mean(pos, axis=0, keepdims=True)
    out.set_positions(pos)
    return out


def load_cnb_isomer_adsorbates(examples_dir: str | Path = "examples") -> dict[str, Atoms]:
    base = Path(examples_dir)
    mapping = {
        "oCNB": base / "oCNB.gjf",
        "mCNB": base / "mCNB.gjf",
        "pCNB": base / "pCNB.gjf",
    }
    out: dict[str, Atoms] = {}
    for key, path in mapping.items():
        atoms = _center_atoms(read(path.as_posix()))
        out[key] = atoms
    return out


def build_pt211_ag4_slab() -> Atoms:
    slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
    pos = np.asarray(slab.get_positions(), dtype=float)
    z = pos[:, 2]
    z_top = float(np.max(z))
    top_ids = np.where(z > z_top - 0.35)[0]
    center_xy = np.mean(pos[top_ids, :2], axis=0)

    edge = 2.90
    base_r = edge / math.sqrt(3.0)
    tetra_h = math.sqrt(2.0 / 3.0) * edge
    z_base = z_top + 2.35

    cluster_pos = np.array(
        [
            [0.0, base_r, 0.0],
            [-0.5 * edge, -0.5 * base_r, 0.0],
            [0.5 * edge, -0.5 * base_r, 0.0],
            [0.0, 0.0, tetra_h],
        ],
        dtype=float,
    )
    cluster_pos[:, 0] += float(center_xy[0])
    cluster_pos[:, 1] += float(center_xy[1])
    cluster_pos[:, 2] += float(z_base)
    cluster = Atoms("Ag4", positions=cluster_pos)
    out = slab + cluster
    out.set_cell(slab.cell)
    out.set_pbc(slab.get_pbc())
    return out


def build_slabs() -> dict[str, Atoms]:
    return {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
        "Pt_fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
        "Pt211Ag4": build_pt211_ag4_slab(),
    }


def build_adsorbates(examples_dir: str | Path = "examples") -> dict[str, Atoms]:
    adsorbates = get_test_adsorbate_cases()
    adsorbates.update(load_cnb_isomer_adsorbates(examples_dir))
    return adsorbates


def default_case_specs() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec("fcc111__NH3", "fcc111", "NH3", "default_production"),
        CaseSpec("Pt_fcc211__CO", "Pt_fcc211", "CO", "default_production"),
        CaseSpec("Pt_fcc211__C2H2", "Pt_fcc211", "C2H2", "default_production"),
        CaseSpec("Pt_fcc211__oCNB", "Pt_fcc211", "oCNB", "expanded_surface"),
        CaseSpec("Pt_fcc211__mCNB", "Pt_fcc211", "mCNB", "expanded_surface"),
        CaseSpec("Pt_fcc211__pCNB", "Pt_fcc211", "pCNB", "expanded_surface"),
        CaseSpec("Pt211Ag4__CO", "Pt211Ag4", "CO", "heterogeneous_support"),
        CaseSpec("Pt211Ag4__C6H6", "Pt211Ag4", "C6H6", "heterogeneous_support"),
    )


def _parse_case_ids(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _build_relax_backend(*, model_path: str | None, device: str, relax_dtype: str, max_edges_per_batch: int) -> MACEBatchRelaxBackend:
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=str(device),
            dtype=str(relax_dtype),
            head_name=DEFAULT_MACE_HEAD_NAME,
            enable_cueq=True,
            strict=True,
            max_edges_per_batch=int(max_edges_per_batch),
        )
    )


def _default_basin_overrides(*, model_path: str | None, device: str) -> dict[str, Any]:
    return {
        "dedup_cluster_method": "greedy",
        "mace_model_path": model_path,
        "mace_device": str(device),
        "mace_dtype": "float64",
        "mace_enable_cueq": False,
        "mace_head_name": DEFAULT_MACE_HEAD_NAME,
        "desorption_min_bonds": 1,
        "energy_window_ev": 2.5,
        "surface_reconstruction_enabled": False,
    }


def _run_default_production_case(
    *,
    case_dir: Path,
    slab: Atoms,
    adsorbate: Atoms,
    relax_backend: MACEBatchRelaxBackend,
    model_path: str | None,
    device: str,
):
    return generate_adsorption_ensemble(
        slab=slab,
        adsorbate=adsorbate,
        work_dir=case_dir,
        placement_mode="anchor_free",
        schedule=make_sampling_schedule("multistage_default"),
        dedup_metric="mace_node_l2",
        signature_mode="provenance",
        basin_overrides=_default_basin_overrides(model_path=model_path, device=device),
        basin_relax_backend=relax_backend,
    )


def _run_expanded_surface_case(
    *,
    case_dir: Path,
    slab: Atoms,
    adsorbate: Atoms,
    relax_backend: MACEBatchRelaxBackend,
    model_path: str | None,
    device: str,
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
            "adaptive_height_fallback": True,
            "adaptive_height_fallback_step": 0.20,
            "adaptive_height_fallback_max_extra": 1.60,
            "adaptive_height_fallback_contact_slack": 0.60,
        },
        basin_overrides=_default_basin_overrides(model_path=model_path, device=device),
    )
    cfg.surface_preprocessor = make_default_surface_preprocessor(
        target_count_mode="adaptive",
        target_surface_fraction=0.25,
    )
    cfg.pre_relax_selection = schedule.pre_relax_selection
    cfg.basin_config.post_relax_selection = schedule.post_relax_selection
    cfg.max_selected_primitives = None
    workflow = run_adsorption_workflow(
        slab=slab,
        adsorbate=adsorbate,
        config=cfg,
        basin_relax_backend=relax_backend,
    )
    readiness = evaluate_adsorption_workflow_readiness(workflow)
    return workflow, readiness


def _run_heterogeneous_support_case(
    *,
    case_dir: Path,
    slab: Atoms,
    adsorbate: Atoms,
    relax_backend: MACEBatchRelaxBackend,
    model_path: str | None,
    device: str,
):
    relaxed_frames, _, _ = relax_backend.relax(
        frames=[slab.copy()],
        maxf=0.05,
        steps=200,
        work_dir=case_dir / "slab_relax",
    )
    slab_relaxed = relaxed_frames[0]
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
            "adaptive_height_fallback": True,
            "adaptive_height_fallback_step": 0.20,
            "adaptive_height_fallback_max_extra": 1.60,
            "adaptive_height_fallback_contact_slack": 0.60,
        },
        basin_overrides=_default_basin_overrides(model_path=model_path, device=device),
    )
    cfg.surface_preprocessor = make_default_surface_preprocessor(
        target_count_mode="off",
        target_surface_fraction=0.25,
    )
    cfg.pre_relax_selection = schedule.pre_relax_selection
    cfg.basin_config.post_relax_selection = schedule.post_relax_selection
    cfg.max_selected_primitives = None
    workflow = run_adsorption_workflow(
        slab=slab_relaxed,
        adsorbate=adsorbate,
        config=cfg,
        basin_relax_backend=relax_backend,
    )
    readiness = evaluate_adsorption_workflow_readiness(workflow)
    return workflow, readiness


def _row_from_generate_result(spec: CaseSpec, result) -> dict[str, Any]:
    return {
        "case": spec.case_id,
        "mode": spec.mode,
        "slab_key": spec.slab_key,
        "adsorbate_key": spec.adsorbate_key,
        "n_surface_atoms": int(result.summary["n_surface_atoms"]),
        "n_basis_primitives": int(result.summary["n_basis_primitives"]),
        "n_pose_frames": int(result.summary["n_pose_frames"]),
        "n_pose_frames_selected_for_basin": int(result.summary["n_pose_frames_selected_for_basin"]),
        "n_basins": int(result.summary["n_basins"]),
        "n_nodes": int(result.summary["n_nodes"]),
        "paper_readiness_score": int(result.readiness.score),
        "paper_readiness_max_score": int(result.readiness.max_score),
        "work_dir": str(result.files["work_dir"]),
    }


def _row_from_workflow(spec: CaseSpec, workflow, readiness, *, work_dir: Path) -> dict[str, Any]:
    return {
        "case": spec.case_id,
        "mode": spec.mode,
        "slab_key": spec.slab_key,
        "adsorbate_key": spec.adsorbate_key,
        "n_surface_atoms": int(workflow.summary["n_surface_atoms"]),
        "n_basis_primitives": int(workflow.summary["n_basis_primitives"]),
        "n_pose_frames": int(workflow.summary["n_pose_frames"]),
        "n_pose_frames_selected_for_basin": int(workflow.summary.get("n_pose_frames_selected_for_basin", workflow.summary["n_pose_frames"])),
        "n_basins": int(workflow.summary["n_basins"]),
        "n_nodes": int(workflow.summary["n_nodes"]),
        "paper_readiness_score": int(readiness.score),
        "paper_readiness_max_score": int(readiness.max_score),
        "work_dir": work_dir.as_posix(),
    }


def _write_rows(rows: list[dict[str, Any]], out_root: Path) -> None:
    summary_json = out_root / "production_case_suite_summary.json"
    summary_csv = out_root / "production_case_suite_summary.csv"
    summary_md = out_root / "production_case_suite_summary.md"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Production Case Suite Summary",
        "",
        "| case | mode | poses | selected | basins | nodes | readiness | work_dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case']}` | `{row['mode']}` | {row['n_pose_frames']} | "
            f"{row['n_pose_frames_selected_for_basin']} | {row['n_basins']} | {row['n_nodes']} | "
            f"{row['paper_readiness_score']}/{row['paper_readiness_max_score']} | `{row['work_dir']}` |"
        )
    summary_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path, model_source = resolve_mace_model_path(str(args.mace_model_path))
    runtime_manifest(
        mace_model_path=model_path,
        model_source=model_source,
        mace_device=str(args.mace_device),
        mace_device_effective=str(args.mace_device),
        mace_dtype="float64",
        out_root=out_root,
    )

    specs = list(default_case_specs())
    if str(args.cases).strip():
        wanted = set(_parse_case_ids(str(args.cases)))
        specs = [spec for spec in specs if spec.case_id in wanted]
        missing = sorted(wanted - {spec.case_id for spec in specs})
        if missing:
            raise KeyError(f"Unknown case ids requested: {missing}")

    slabs = build_slabs()
    adsorbates = build_adsorbates(args.examples_dir)
    relax_backend = _build_relax_backend(
        model_path=model_path,
        device=str(args.mace_device),
        relax_dtype=str(args.relax_dtype),
        max_edges_per_batch=int(args.max_edges_per_batch),
    )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        case_dir = out_root / spec.case_id
        slab = slabs[spec.slab_key]
        adsorbate = adsorbates[spec.adsorbate_key]
        if spec.mode == "default_production":
            result = _run_default_production_case(
                case_dir=case_dir,
                slab=slab,
                adsorbate=adsorbate,
                relax_backend=relax_backend,
                model_path=model_path,
                device=str(args.mace_device),
            )
            rows.append(_row_from_generate_result(spec, result))
            continue
        if spec.mode == "expanded_surface":
            workflow, readiness = _run_expanded_surface_case(
                case_dir=case_dir,
                slab=slab,
                adsorbate=adsorbate,
                relax_backend=relax_backend,
                model_path=model_path,
                device=str(args.mace_device),
            )
            rows.append(_row_from_workflow(spec, workflow, readiness, work_dir=case_dir))
            continue
        if spec.mode == "heterogeneous_support":
            workflow, readiness = _run_heterogeneous_support_case(
                case_dir=case_dir,
                slab=slab,
                adsorbate=adsorbate,
                relax_backend=relax_backend,
                model_path=model_path,
                device=str(args.mace_device),
            )
            rows.append(_row_from_workflow(spec, workflow, readiness, work_dir=case_dir))
            continue
        raise ValueError(f"Unsupported mode: {spec.mode}")

    _write_rows(rows, out_root)
    print((out_root / "production_case_suite_summary.json").as_posix())
    return out_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/production_case_suite_v1")
    parser.add_argument("--examples-dir", type=str, default="examples")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--relax-dtype", type=str, default="float32")
    parser.add_argument("--max-edges-per-batch", type=int, default=100000)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
