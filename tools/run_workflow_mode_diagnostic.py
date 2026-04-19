from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import generate_adsorption_ensemble, make_sampling_schedule
from tests.chemistry_cases import get_test_adsorbate_cases
from tools.run_ase_autoadsorbate_crosscheck import build_slab_suite


DEFAULT_CASES = ("TiO2_110__NH3",)

DEFAULT_MODES: tuple[tuple[str, str, str, bool], ...] = (
    ("anchor_free_default", "anchor_free", "multistage_default", False),
    ("anchor_free_default_exhaustive", "anchor_free", "multistage_default", True),
    ("anchor_free_no_selection_exhaustive", "anchor_free", "no_selection", True),
    ("anchor_aware_default", "anchor_aware", "multistage_default", False),
    ("anchor_aware_default_exhaustive", "anchor_aware", "multistage_default", True),
    ("anchor_aware_no_selection_exhaustive", "anchor_aware", "no_selection", True),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_cases(raw: str) -> list[tuple[str, str]]:
    out = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        slab_name, ads_name = token.split("__", 1)
        out.append((str(slab_name), str(ads_name)))
    return out


def _make_relax_backend(model_path: str, device: str, dtype: str, head_name: str) -> MACEBatchRelaxBackend:
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=str(model_path),
            device=str(device),
            dtype=str(dtype),
            max_edges_per_batch=20000,
            head_name=str(head_name),
            enable_cueq=True,
            strict=True,
        )
    )


def run(args: argparse.Namespace) -> Path:
    slabs = build_slab_suite()
    adsorbates = get_test_adsorbate_cases()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    relax_backend = _make_relax_backend(
        model_path=str(args.mace_model_path),
        device=str(args.mace_device),
        dtype=str(args.mace_dtype),
        head_name=str(args.mace_head_name),
    )
    cases = _parse_cases(str(args.cases)) if str(args.cases).strip() else _parse_cases(",".join(DEFAULT_CASES))

    rows: list[dict[str, Any]] = []
    for slab_name, ads_name in cases:
        slab = slabs[slab_name]
        adsorbate = adsorbates[ads_name]
        for mode_name, placement_mode, schedule_preset, exhaustive in DEFAULT_MODES:
            case_dir = out_root / slab_name / ads_name / mode_name
            schedule = make_sampling_schedule(schedule_preset, exhaustive_pose_sampling=bool(exhaustive))
            result = generate_adsorption_ensemble(
                slab=slab,
                adsorbate=adsorbate.copy(),
                work_dir=case_dir,
                placement_mode=str(placement_mode),
                schedule=schedule,
                dedup_metric="binding_surface_distance",
                signature_mode="provenance",
                basin_overrides={
                    "mace_model_path": str(args.mace_model_path),
                    "mace_device": str(args.mace_device),
                    "mace_dtype": str(args.mace_dtype),
                    "mace_head_name": str(args.mace_head_name),
                    "final_basin_merge_metric": "mace_node_l2",
                    "final_basin_merge_node_l2_threshold": 0.20,
                    "final_basin_merge_cluster_method": "hierarchical",
                },
                basin_relax_backend=relax_backend,
            )
            row = {
                "case": f"{slab_name}__{ads_name}",
                "mode": str(mode_name),
                "placement_mode": str(placement_mode),
                "schedule_name": str(schedule.name),
                "exhaustive_pose_sampling": bool(exhaustive),
                "work_dir": case_dir.as_posix(),
                "n_surface_atoms": int(result.summary["n_surface_atoms"]),
                "n_basis_primitives": int(result.summary["n_basis_primitives"]),
                "n_pose_frames": int(result.summary["n_pose_frames"]),
                "n_pose_frames_selected_for_basin": int(result.summary["n_pose_frames_selected_for_basin"]),
                "n_basins": int(result.summary["n_basins"]),
                "n_nodes": int(result.summary["n_nodes"]),
            }
            basins_json = Path(result.files.get("basins_json", ""))
            if basins_json.exists():
                payload = json.loads(basins_json.read_text(encoding="utf-8"))
                counts: dict[str, int] = {}
                for rej in payload.get("rejected", []):
                    reason = str(rej.get("reason", "")).strip()
                    if reason:
                        counts[reason] = counts.get(reason, 0) + 1
                row["rejected_reason_counts"] = counts
            else:
                row["rejected_reason_counts"] = {}
            rows.append(row)
            _write_json(case_dir / "workflow_mode_case_summary.json", row)

    summary = {
        "out_root": out_root.as_posix(),
        "cases": [f"{slab}__{ads}" for slab, ads in cases],
        "modes": [
            {
                "mode": str(name),
                "placement_mode": str(placement),
                "schedule_preset": str(schedule),
                "exhaustive_pose_sampling": bool(exhaustive),
            }
            for name, placement, schedule, exhaustive in DEFAULT_MODES
        ],
        "rows": rows,
    }
    _write_json(out_root / "workflow_mode_diagnostic_summary.json", summary)
    md_lines = [
        "# Workflow Mode Diagnostic",
        "",
    ]
    for case in sorted(set(r["case"] for r in rows)):
        md_lines.append(f"## {case}")
        md_lines.append("")
        for row in [r for r in rows if r["case"] == case]:
            md_lines.append(
                f"- {row['mode']}: basins={row['n_basins']}, poses={row['n_pose_frames']}, "
                f"selected={row['n_pose_frames_selected_for_basin']}, rejected={row['rejected_reason_counts']}"
            )
        md_lines.append("")
    (out_root / "workflow_mode_diagnostic_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print((out_root / "workflow_mode_diagnostic_summary.json").as_posix())
    print((out_root / "workflow_mode_diagnostic_summary.md").as_posix())
    return out_root / "workflow_mode_diagnostic_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/workflow_mode_diagnostic_20260405")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--mace-head-name", type=str, default="omat_pbe")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
