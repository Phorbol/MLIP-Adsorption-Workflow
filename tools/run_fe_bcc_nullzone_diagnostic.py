from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import generate_adsorption_ensemble, make_sampling_schedule
from tools.run_miller_monodentate_matrix import build_miller_metal_slab_suite, build_monodentate_suite


DEFAULT_CASES = (
    "Fe_bcc100__CO",
    "Fe_bcc100__H2O",
    "Fe_bcc110__H2O",
    "Fe_bcc111__CO",
    "Fe_bcc111__H2O",
)


RUN_MODES: tuple[tuple[str, str, bool], ...] = (
    ("default", "multistage_default", False),
    ("default_exhaustive", "multistage_default", True),
    ("no_selection_exhaustive", "no_selection", True),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _make_relax_backend(model_path: str, device: str, dtype: str, enable_cueq: bool) -> MACEBatchRelaxBackend:
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=str(model_path),
            device=str(device),
            dtype=str(dtype),
            max_edges_per_batch=20000,
            head_name="omat_pbe",
            enable_cueq=bool(enable_cueq),
            strict=True,
        )
    )


def _load_basins_payload(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_case_mode(
    *,
    case_name: str,
    slab,
    adsorbate,
    mode_name: str,
    schedule_preset: str,
    exhaustive_pose_sampling: bool,
    out_root: Path,
    relax_backend: MACEBatchRelaxBackend,
) -> dict[str, Any]:
    work_dir = out_root / case_name / mode_name
    schedule = make_sampling_schedule(
        schedule_preset,
        exhaustive_pose_sampling=bool(exhaustive_pose_sampling),
    )
    result = generate_adsorption_ensemble(
        slab=slab,
        adsorbate=adsorbate.copy(),
        work_dir=work_dir,
        placement_mode="anchor_free",
        schedule=schedule,
        dedup_metric="binding_surface_distance",
        signature_mode="provenance",
        basin_overrides={
            "mace_model_path": "/root/.cache/mace/mace-omat-0-small.model",
            "mace_device": "cuda",
            "mace_dtype": "float32",
            "mace_head_name": "omat_pbe",
            "final_basin_merge_metric": "mace_node_l2",
            "final_basin_merge_node_l2_threshold": 0.20,
            "final_basin_merge_cluster_method": "hierarchical",
        },
        basin_relax_backend=relax_backend,
    )
    basins_payload = _load_basins_payload(result.files.get("basins_json"))
    rejected_reason_counts = Counter(str(row.get("reason", "")) for row in basins_payload.get("rejected", []))
    basin_site_labels = Counter(str(row.get("site_label", "")) for row in basins_payload.get("basins", []))
    return {
        "case": str(case_name),
        "mode": str(mode_name),
        "schedule_name": str(result.summary["schedule"]["name"]),
        "exhaustive_pose_sampling": bool(exhaustive_pose_sampling),
        "work_dir": str(work_dir.as_posix()),
        "n_basis_primitives": int(result.summary["n_basis_primitives"]),
        "n_pose_frames": int(result.summary["n_pose_frames"]),
        "n_pose_frames_selected_for_basin": int(result.summary["n_pose_frames_selected_for_basin"]),
        "n_basins": int(result.summary["n_basins"]),
        "rejected_reason_counts": {str(k): int(v) for k, v in sorted(rejected_reason_counts.items()) if str(k)},
        "basin_site_label_counts": {str(k): int(v) for k, v in sorted(basin_site_labels.items()) if str(k)},
        "files": dict(result.files),
        "readiness": dict(result.readiness.summary),
    }


def run(args: argparse.Namespace) -> Path:
    cases = [str(x).strip() for x in str(args.cases).split(",") if str(x).strip()] or list(DEFAULT_CASES)
    slabs = build_miller_metal_slab_suite()
    molecules = build_monodentate_suite()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    relax_backend = _make_relax_backend(
        model_path=str(args.mace_model_path),
        device=str(args.mace_device),
        dtype=str(args.mace_dtype),
        enable_cueq=bool(not args.disable_cueq),
    )

    rows: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for case_name in cases:
        slab_name, ads_name = str(case_name).split("__", 1)
        slab = slabs[slab_name]
        adsorbate = molecules[ads_name]
        case_rows = []
        for mode_name, schedule_preset, exhaustive in RUN_MODES:
            row = _run_case_mode(
                case_name=case_name,
                slab=slab,
                adsorbate=adsorbate,
                mode_name=mode_name,
                schedule_preset=schedule_preset,
                exhaustive_pose_sampling=bool(exhaustive),
                out_root=out_root,
                relax_backend=relax_backend,
            )
            rows.append(row)
            case_rows.append(row)
            _write_json(out_root / case_name / mode_name / "nullzone_case_summary.json", row)
        by_case[str(case_name)] = case_rows

    case_diagnosis = []
    for case_name, case_rows in by_case.items():
        all_zero = all(int(row["n_basins"]) == 0 for row in case_rows)
        any_pose_gain = any(int(row["n_pose_frames"]) > int(case_rows[0]["n_pose_frames"]) for row in case_rows[1:])
        dominant_reasons = Counter()
        for row in case_rows:
            dominant_reasons.update({str(k): int(v) for k, v in row["rejected_reason_counts"].items()})
        case_diagnosis.append(
            {
                "case": str(case_name),
                "all_modes_zero_basins": bool(all_zero),
                "any_pose_budget_gain_vs_default": bool(any_pose_gain),
                "aggregate_rejected_reason_counts": {str(k): int(v) for k, v in sorted(dominant_reasons.items())},
            }
        )

    payload = {
        "out_root": out_root.as_posix(),
        "mace_model_path": str(args.mace_model_path),
        "mace_device": str(args.mace_device),
        "mace_dtype": str(args.mace_dtype),
        "enable_cueq": bool(not args.disable_cueq),
        "cases": list(cases),
        "run_modes": [
            {"mode": str(mode), "schedule_preset": str(preset), "exhaustive_pose_sampling": bool(exhaustive)}
            for mode, preset, exhaustive in RUN_MODES
        ],
        "rows": rows,
        "case_diagnosis": case_diagnosis,
    }
    _write_json(out_root / "fe_bcc_nullzone_diagnostic.json", payload)

    md_lines = [
        "# Fe BCC Null-Zone Diagnostic",
        "",
        f"- Cases: {len(cases)}",
        f"- Run modes: {len(RUN_MODES)}",
        "",
        "## Case Diagnosis",
        "",
    ]
    for row in case_diagnosis:
        md_lines.append(
            f"- {row['case']}: all_modes_zero_basins={row['all_modes_zero_basins']}, "
            f"any_pose_budget_gain_vs_default={row['any_pose_budget_gain_vs_default']}, "
            f"aggregate_rejected_reason_counts={row['aggregate_rejected_reason_counts']}"
        )
    (out_root / "fe_bcc_nullzone_diagnostic.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print((out_root / "fe_bcc_nullzone_diagnostic.json").as_posix())
    print((out_root / "fe_bcc_nullzone_diagnostic.md").as_posix())
    return out_root / "fe_bcc_nullzone_diagnostic.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/nullzone_diagnostic/fe_bcc_20260405")
    parser.add_argument("--cases", type=str, default=",".join(DEFAULT_CASES))
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--disable-cueq", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
