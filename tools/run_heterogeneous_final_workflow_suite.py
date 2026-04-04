from __future__ import annotations

import argparse
import contextlib
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read

from adsorption_ensemble.basin import BasinConfig, run_named_basin_ablation
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import evaluate_adsorption_workflow_readiness, generate_adsorption_ensemble, make_sampling_schedule
from tests.chemistry_cases import get_test_adsorbate_cases
from tools.run_ase_autoadsorbate_crosscheck import build_slab_suite
from tools.run_autoresearch_artifact_suite import infer_mace_head_name, resolve_mace_model_path, runtime_manifest


DEFAULT_CASES = (
    "CuNi_fcc111_alloy__CO",
    "MgO_100__NH3",
    "Pt_fcc111_vacancy__CO",
    "Pt_fcc111_adatom__CO",
    "Pt_fcc111_cluster_interface__C6H6",
)

DIAGNOSTIC_CASES = (
    "TiO2_110__NH3",
)


def _parse_cases(raw: str) -> list[tuple[str, str]]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    out = []
    for token in tokens:
        if "__" not in token:
            raise ValueError(f"Case must use slab__adsorbate form: {token}")
        slab_name, ads_name = token.split("__", 1)
        out.append((str(slab_name), str(ads_name)))
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_rejected_reason_counts(case_dir: Path) -> dict[str, int]:
    basins_json = case_dir / "basins.json"
    if not basins_json.exists():
        return {}
    try:
        payload = json.loads(basins_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rejected = payload.get("rejected", [])
    return dict(Counter(str(r.get("reason", "")) for r in rejected))


def _build_relax_backend(*, model_path: str | None, device: str, dtype: str) -> MACEBatchRelaxBackend:
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=str(device),
            dtype=str(dtype),
            max_edges_per_batch=20000,
            head_name=infer_mace_head_name(model_path),
            enable_cueq=True,
            strict=True,
        )
    )


def _default_ablation_configs(*, model_path: str | None, device: str, dtype: str) -> dict[str, BasinConfig]:
    common = {
        "relax_maxf": 0.1,
        "relax_steps": 80,
        "energy_window_ev": 2.5,
        "desorption_min_bonds": 1,
        "binding_tau": 1.15,
        "work_dir": None,
        "mace_model_path": model_path,
        "mace_device": str(device),
        "mace_dtype": str(dtype),
    }
    return {
        "signature_only": BasinConfig(
            dedup_metric="signature_only",
            signature_mode="provenance",
            dedup_cluster_method="greedy",
            **common,
        ),
        "rmsd": BasinConfig(
            dedup_metric="rmsd",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "binding_surface_greedy": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="greedy",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            **common,
        ),
        "mace_node_l2_0p10": BasinConfig(
            dedup_metric="mace_node_l2",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.10,
            **common,
        ),
        "mace_node_l2_0p20": BasinConfig(
            dedup_metric="mace_node_l2",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.20,
            **common,
        ),
    }


def _load_relaxed_frames(case_dir: Path) -> list:
    candidates = (
        case_dir / "basin_work" / "post_relax_selected.extxyz",
        case_dir / "basin_work" / "relax" / "relaxed_stream.extxyz",
    )
    for path in candidates:
        if path.exists():
            return list(read(path.as_posix(), index=":"))
    raise FileNotFoundError(f"No relaxed-frame artifact found for case directory: {case_dir}")


def _write_review_index(out_root: Path, rows: list[dict[str, Any]]) -> Path:
    lines = [
        "# Heterogeneous Final Workflow Review Index",
        "",
        f"Root: `{out_root.as_posix()}`",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        case_dir = Path(str(row["work_dir"]))
        lines.extend(
            [
                f"### {row['case']}",
                "",
                f"- work_dir: `{case_dir.as_posix()}`",
                f"- sites: `{(case_dir / 'sites.png').as_posix()}`",
                f"- site_dictionary: `{(case_dir / 'site_dictionary.json').as_posix()}`",
                f"- pose_pool_selected: `{(case_dir / 'pose_pool_selected.extxyz').as_posix()}`",
                f"- basins_extxyz: `{(case_dir / 'basins.extxyz').as_posix()}`",
                f"- basins_json: `{(case_dir / 'basins.json').as_posix()}`",
                f"- basin_dictionary: `{(case_dir / 'basin_dictionary.json').as_posix()}`",
                f"- nodes_json: `{(case_dir / 'nodes.json').as_posix()}`",
                f"- workflow_summary: `{(case_dir / 'workflow_summary.json').as_posix()}`",
                f"- ablation_json: `{(case_dir / 'heterogeneous_final_ablation.json').as_posix()}`",
                "",
            ]
        )
    out_path = out_root / "review_index.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_path


def _write_report(out_root: Path, rows: list[dict[str, Any]], *, schedule_name: str) -> Path:
    lines = [
        "# Heterogeneous Final Workflow Suite Report",
        "",
        "## Workflow",
        "",
        f"- placement_mode: `anchor_free`",
        f"- schedule: `{schedule_name}`",
        "- pre_relax_selection: fixed FPS `k=24`",
        "- post_relax_selection: energy window + RMSD window",
        "- final_dedup_stage1: `binding_surface_distance + greedy`",
        "- final_dedup_stage1_params: `nearest_k=8`, `threshold=0.30`, `binding_only`, `relative=False`, `rmsd_gate=0.25`",
        "- final_dedup_stage2: `pure_mace basin merge + hierarchical`",
        "- final_dedup_stage2_params: `node_l2_threshold=0.20`, `mean_atom`",
        "",
        "## Case Summary",
        "",
        "| case | surface_atoms | basis_primitives | poses | selected | basins | nodes | readiness | ablation(binding_surface / rmsd / mace0.10 / mace0.20) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        ab = row["ablation"]
        lines.append(
            f"| `{row['case']}` | {row['n_surface_atoms']} | {row['n_basis_primitives']} | "
            f"{row['n_pose_frames']} | {row['n_pose_frames_selected_for_basin']} | {row['workflow_n_basins']} | "
            f"{row['n_nodes']} | {row['paper_readiness_score']}/{row['paper_readiness_max_score']} | "
            f"{ab['binding_surface_greedy_n_basins']} / {ab['rmsd_n_basins']} / {ab['mace_node_l2_0p10_n_basins']} / {ab['mace_node_l2_0p20_n_basins']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This suite extends the frozen default workflow from terrace/vicinal cases to alloy, oxide, defect, and cluster-interface surfaces.",
            "- Each case preserves the full artifact chain: site PNGs, site dictionaries, selected pose pool, `basins.extxyz`, `basins.json`, `basin_dictionary.json`, and `nodes.json`.",
            "",
        ]
    )
    out_path = out_root / "heterogeneous_final_workflow_suite_report.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_path


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _parse_cases(str(args.cases)) if str(args.cases).strip() else _parse_cases(",".join(DEFAULT_CASES))
    if bool(args.include_diagnostic_cases):
        cases.extend(_parse_cases(",".join(DIAGNOSTIC_CASES)))
    slabs = build_slab_suite()
    adsorbates = get_test_adsorbate_cases()

    model_path, model_source = resolve_mace_model_path(str(args.mace_model_path))
    runtime_manifest(
        mace_model_path=model_path,
        model_source=model_source,
        mace_device=str(args.mace_device),
        mace_device_effective=str(args.mace_device),
        mace_dtype=str(args.mace_dtype),
        out_root=out_root,
    )
    relax_backend = _build_relax_backend(model_path=model_path, device=str(args.mace_device), dtype=str(args.mace_dtype))
    schedule = make_sampling_schedule(str(args.schedule_preset))
    ablation_cfgs = _default_ablation_configs(model_path=model_path, device=str(args.mace_device), dtype=str(args.mace_dtype))

    rows: list[dict[str, Any]] = []
    for slab_name, ads_name in cases:
        if slab_name not in slabs:
            raise KeyError(f"Unknown slab case: {slab_name}")
        if ads_name not in adsorbates:
            raise KeyError(f"Unknown adsorbate case: {ads_name}")
        case_dir = out_root / slab_name / ads_name
        if case_dir.exists() and not bool(args.rerun_existing):
            summary_path = case_dir / "heterogeneous_final_summary.json"
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue

        result = generate_adsorption_ensemble(
            slab=slabs[slab_name],
            adsorbate=adsorbates[ads_name],
            work_dir=case_dir,
            placement_mode="anchor_free",
            schedule=schedule,
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            basin_overrides={
                "dedup_cluster_method": "greedy",
                "surface_descriptor_threshold": 0.30,
                "surface_descriptor_nearest_k": 8,
                "surface_descriptor_atom_mode": "binding_only",
                "surface_descriptor_relative": False,
                "surface_descriptor_rmsd_gate": 0.25,
                "final_basin_merge_metric": "mace_node_l2",
                "final_basin_merge_node_l2_threshold": 0.20,
                "final_basin_merge_cluster_method": "hierarchical",
                "desorption_min_bonds": 1,
                "energy_window_ev": 2.5,
                "mace_model_path": model_path,
                "mace_device": str(args.mace_device),
                "mace_dtype": str(args.mace_dtype),
                "mace_head_name": infer_mace_head_name(model_path),
            },
            basin_relax_backend=relax_backend,
        )
        readiness = evaluate_adsorption_workflow_readiness(result.workflow)
        relaxed_frames = _load_relaxed_frames(case_dir)
        ablation_log = case_dir / "heterogeneous_final_ablation.stdout_stderr.log"
        with ablation_log.open("a", encoding="utf-8") as capture_stream:
            with contextlib.redirect_stdout(capture_stream), contextlib.redirect_stderr(capture_stream):
                ablation = run_named_basin_ablation(
                    frames=relaxed_frames,
                    slab_ref=slabs[slab_name],
                    adsorbate_ref=adsorbates[ads_name],
                    slab_n=len(slabs[slab_name]),
                    normal_axis=2,
                    configs=ablation_cfgs,
                    relax_backend=None,
                )
        _write_json(case_dir / "heterogeneous_final_ablation.json", ablation)
        row = {
            "slab": str(slab_name),
            "adsorbate": str(ads_name),
            "case": f"{slab_name}__{ads_name}",
            "work_dir": case_dir.as_posix(),
            "n_surface_atoms": int(result.summary["n_surface_atoms"]),
            "n_basis_primitives": int(result.summary["n_basis_primitives"]),
            "n_pose_frames": int(result.summary["n_pose_frames"]),
            "n_pose_frames_selected_for_basin": int(
                result.summary.get("n_pose_frames_selected_for_basin", result.summary["n_pose_frames"])
            ),
            "workflow_n_basins": int(result.summary["n_basins"]),
            "n_nodes": int(result.summary["n_nodes"]),
            "paper_readiness_score": int(readiness.score),
            "paper_readiness_max_score": int(readiness.max_score),
            "rejected_reason_counts": _load_rejected_reason_counts(case_dir),
            "ablation": {
                "signature_only_n_basins": int(ablation["configs"]["signature_only"]["n_basins"]),
                "rmsd_n_basins": int(ablation["configs"]["rmsd"]["n_basins"]),
                "binding_surface_greedy_n_basins": int(ablation["configs"]["binding_surface_greedy"]["n_basins"]),
                "mace_node_l2_0p10_n_basins": int(ablation["configs"]["mace_node_l2_0p10"]["n_basins"]),
                "mace_node_l2_0p20_n_basins": int(ablation["configs"]["mace_node_l2_0p20"]["n_basins"]),
            },
        }
        _write_json(case_dir / "heterogeneous_final_summary.json", row)
        rows.append(row)

    summary_json = out_root / "heterogeneous_final_workflow_suite_summary.json"
    summary_csv = out_root / "heterogeneous_final_workflow_suite_summary.csv"
    _write_json(summary_json, rows)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "slab",
            "adsorbate",
            "case",
            "work_dir",
            "n_surface_atoms",
            "n_basis_primitives",
            "n_pose_frames",
            "n_pose_frames_selected_for_basin",
            "workflow_n_basins",
            "n_nodes",
            "paper_readiness_score",
            "paper_readiness_max_score",
            "rejected_reason_counts",
            "ablation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = _write_report(out_root, rows, schedule_name=str(schedule.name))
    review_index_path = _write_review_index(out_root, rows)
    payload = {
        "summary_json": summary_json.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "report_md": report_path.as_posix(),
        "review_index_md": review_index_path.as_posix(),
        "n_cases": len(rows),
    }
    _write_json(out_root / "heterogeneous_final_workflow_suite_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return summary_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=str,
        default="artifacts/autoresearch/heterogeneous_final_workflow_suite_20260404",
    )
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--schedule-preset", type=str, default="multistage_default")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--include-diagnostic-cases", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
