from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def reference_source_of_row(row: dict[str, Any]) -> str:
    source = str(row.get("ase_manual", {}).get("reference_source", "")).strip()
    if source:
        return source
    n_input_sites = int(row.get("ase_manual", {}).get("n_input_sites", 0))
    if n_input_sites > 0:
        return "ase_adsorbate_info"
    return ""


def classify_case(row: dict[str, Any]) -> str:
    n_input_sites = int(row.get("ase_manual", {}).get("n_input_sites", 0))
    n_manual = int(row.get("ase_manual", {}).get("n_basins", 0))
    n_ours = int(row.get("ours", {}).get("n_basins", 0))
    reference_source = reference_source_of_row(row)
    recall_raw = row.get("overlap", {}).get("manual_recall_by_ours", None)
    recall = None if recall_raw is None else float(recall_raw)
    if n_input_sites <= 0 and not reference_source:
        return "manual_reference_missing"
    if n_manual == 0 and n_ours == 0:
        return "both_zero"
    if n_manual == 0 and n_ours > 0:
        return "manual_zero_ours_positive"
    if n_manual > 0 and n_ours == 0:
        return "manual_positive_ours_zero"
    if recall is not None and abs(recall - 1.0) <= 1e-12:
        return "full_recall"
    return "partial_recall"


def _nullzone_map(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("case_diagnosis", [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case = str(row.get("case", "")).strip()
        if case:
            out[case] = dict(row)
    return out


def build_audit(payload: dict[str, Any], *, nullzone_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = payload.get("rows", [])
    nullzone_by_case = _nullzone_map(nullzone_payload)
    case_rows = []
    status_counter: Counter[str] = Counter()
    for row in rows:
        status = classify_case(row)
        nullzone_row = nullzone_by_case.get(str(row.get("case", "")).strip())
        if status == "both_zero" and isinstance(nullzone_row, dict) and bool(nullzone_row.get("all_modes_zero_basins", False)):
            status = "robust_null_zone"
        status_counter[str(status)] += 1
        sentinel = row.get("sentinel_audit", {})
        case_rows.append(
            {
                "case": str(row.get("case", "")),
                "slab": str(row.get("slab", "")),
                "adsorbate": str(row.get("adsorbate", "")),
                "comparison_status": str(status),
                "reference_source": reference_source_of_row(row),
                "n_basis_primitives": int(row.get("ours", {}).get("n_basis_primitives", 0)),
                "n_pose_frames": int(row.get("ours", {}).get("n_pose_frames", 0)),
                "n_selected_pose_frames": int(row.get("ours", {}).get("n_pose_frames_selected_for_basin", 0)),
                "ours_n_basins": int(row.get("ours", {}).get("n_basins", 0)),
                "manual_n_input_sites": int(row.get("ase_manual", {}).get("n_input_sites", 0)),
                "manual_n_basins": int(row.get("ase_manual", {}).get("n_basins", 0)),
                "manual_recall_by_ours": (
                    None
                    if row.get("overlap", {}).get("manual_recall_by_ours", None) is None
                    else float(row.get("overlap", {}).get("manual_recall_by_ours"))
                ),
                "manual_rejected_reason_counts": dict(row.get("ase_manual", {}).get("rejected_reason_counts", {})),
                "nullzone_diagnostic": (None if nullzone_row is None else dict(nullzone_row)),
                "sentinel_interpretation": str(sentinel.get("interpretation", "")) if sentinel else "",
                "sentinel_coordination": sentinel.get("final_binding_environment", {}).get("coordination") if sentinel else None,
            }
        )

    manual_defined = [r for r in case_rows if int(r["manual_n_input_sites"]) > 0]
    manual_positive = [r for r in case_rows if int(r["manual_n_basins"]) > 0]
    recall_defined = [
        r
        for r in case_rows
        if r.get("manual_recall_by_ours", None) is not None and int(r["manual_n_basins"]) > 0
    ]
    both_zero = [r for r in case_rows if str(r["comparison_status"]) == "both_zero"]
    robust_null_zone = [r for r in case_rows if str(r["comparison_status"]) == "robust_null_zone"]
    manual_missing = [r for r in case_rows if str(r["comparison_status"]) == "manual_reference_missing"]
    sentinel_cases = [r for r in case_rows if str(r["sentinel_interpretation"]).strip()]
    source_counts: Counter[str] = Counter(str(r.get("reference_source", "")).strip() or "missing" for r in case_rows)
    ase_reference_cases = [r for r in case_rows if str(r.get("reference_source", "")) == "ase_adsorbate_info"]
    fallback_reference_cases = [r for r in case_rows if str(r.get("reference_source", "")) == "primitive_basis_fallback"]
    ase_positive = [r for r in ase_reference_cases if int(r["manual_n_basins"]) > 0]
    fallback_positive = [r for r in fallback_reference_cases if int(r["manual_n_basins"]) > 0]
    return {
        "source_out_root": payload.get("out_root", ""),
        "n_cases": int(len(case_rows)),
        "status_counts": {str(k): int(v) for k, v in sorted(status_counter.items())},
        "reference_source_counts": {str(k): int(v) for k, v in sorted(source_counts.items())},
        "manual_defined_cases": int(len(manual_defined)),
        "manual_positive_cases": int(len(manual_positive)),
        "recall_defined_cases": int(len(recall_defined)),
        "manual_defined_mean_recall": (
            None
            if not recall_defined
            else float(sum(float(r["manual_recall_by_ours"]) for r in recall_defined) / len(recall_defined))
        ),
        "manual_positive_mean_recall": float(
            sum(float(r["manual_recall_by_ours"]) for r in manual_positive) / max(1, len(manual_positive))
        ),
        "manual_positive_full_recall_cases": int(
            sum(abs(float(r["manual_recall_by_ours"]) - 1.0) <= 1e-12 for r in manual_positive)
        ),
        "ase_reference_cases": int(len(ase_reference_cases)),
        "ase_reference_positive_cases": int(len(ase_positive)),
        "ase_reference_positive_mean_recall": float(
            sum(float(r["manual_recall_by_ours"]) for r in ase_positive) / max(1, len(ase_positive))
        ),
        "fallback_reference_cases": int(len(fallback_reference_cases)),
        "fallback_reference_positive_cases": int(len(fallback_positive)),
        "fallback_reference_positive_mean_recall": float(
            sum(float(r["manual_recall_by_ours"]) for r in fallback_positive) / max(1, len(fallback_positive))
        ),
        "both_zero_cases": [dict(r) for r in both_zero],
        "robust_null_zone_cases": [dict(r) for r in robust_null_zone],
        "manual_reference_missing_cases": [dict(r) for r in manual_missing],
        "sentinel_cases": [dict(r) for r in sentinel_cases],
        "case_rows": case_rows,
    }


def write_audit(audit: dict[str, Any], out_root: Path) -> tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "miller_monodentate_validation_audit.json"
    md_path = out_root / "miller_monodentate_validation_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Miller Monodentate Validation Audit",
        "",
        f"- Cases: {audit['n_cases']}",
        f"- Status counts: {audit['status_counts']}",
        f"- Reference sources: {audit['reference_source_counts']}",
        f"- Recall-defined cases: {audit['recall_defined_cases']}",
        f"- Manual-defined mean recall: {'NA' if audit['manual_defined_mean_recall'] is None else f'{audit['manual_defined_mean_recall']:.3f}'}",
        f"- Manual-positive mean recall: {audit['manual_positive_mean_recall']:.3f}",
        f"- Manual-positive full recall: {audit['manual_positive_full_recall_cases']} / {audit['manual_positive_cases']}",
        f"- ASE-positive mean recall: {audit['ase_reference_positive_mean_recall']:.3f}",
        f"- Primitive-fallback positive mean recall: {audit['fallback_reference_positive_mean_recall']:.3f}",
        "",
        "## Cases",
        "",
    ]
    for row in audit["case_rows"]:
        recall_text = "NA" if row["manual_recall_by_ours"] is None else f"{float(row['manual_recall_by_ours']):.3f}"
        lines.append(
            f"- {row['case']}: source={row['reference_source'] or 'missing'}, status={row['comparison_status']}, ours={row['ours_n_basins']}, "
            f"manual={row['manual_n_basins']}, recall={recall_text}, "
            f"sentinel={row['sentinel_interpretation'] or 'none'}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-json", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--nullzone-json", type=str, default="")
    args = parser.parse_args()

    payload = json.loads(Path(args.validation_json).read_text(encoding="utf-8"))
    nullzone_payload = None
    if str(args.nullzone_json).strip():
        nullzone_payload = json.loads(Path(args.nullzone_json).read_text(encoding="utf-8"))
    audit = build_audit(payload, nullzone_payload=nullzone_payload)
    json_path, md_path = write_audit(audit, Path(args.out_root))
    print(json_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
