from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_case(summary_path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_json(summary_path)
    return {str(row["case"]): dict(row) for row in rows}


def _case_dir_from_row(row: dict[str, Any]) -> Path:
    return Path(str(row["work_dir"]))


def _pre_relax_diag(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "pre_relax_selection.json"
    if not path.exists():
        return {}
    return dict(_load_json(path))


def _post_relax_diag(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "basin_work" / "post_relax_selection.json"
    if not path.exists():
        return {}
    return dict(_load_json(path))


def _build_case_row(case: str, base: dict[str, Any], other: dict[str, Any], *, base_label: str, other_label: str) -> dict[str, Any]:
    base_dir = _case_dir_from_row(base)
    other_dir = _case_dir_from_row(other)
    base_pre = _pre_relax_diag(base_dir)
    other_pre = _pre_relax_diag(other_dir)
    base_post = _post_relax_diag(base_dir)
    other_post = _post_relax_diag(other_dir)
    return {
        "case": str(case),
        "slab": str(base["slab"]),
        "adsorbate": str(base["adsorbate"]),
        f"{base_label}_n_pose_frames": int(base["n_pose_frames"]),
        f"{other_label}_n_pose_frames": int(other["n_pose_frames"]),
        f"{base_label}_n_pose_frames_selected_for_basin": int(base["n_pose_frames_selected_for_basin"]),
        f"{other_label}_n_pose_frames_selected_for_basin": int(other["n_pose_frames_selected_for_basin"]),
        f"{base_label}_workflow_n_basins": int(base["workflow_n_basins"]),
        f"{other_label}_workflow_n_basins": int(other["workflow_n_basins"]),
        f"{base_label}_ablation_n_frames": int(base["ablation_n_frames"]),
        f"{other_label}_ablation_n_frames": int(other["ablation_n_frames"]),
        f"{base_label}_pre_strategy": str(base_pre.get("strategy", "")),
        f"{other_label}_pre_strategy": str(other_pre.get("strategy", "")),
        f"{base_label}_pre_n_selected": int(base_pre.get("n_selected", base["n_pose_frames_selected_for_basin"])),
        f"{other_label}_pre_n_selected": int(other_pre.get("n_selected", other["n_pose_frames_selected_for_basin"])),
        f"{base_label}_pre_stopped_by_convergence": bool(base_pre.get("stopped_by_convergence", False)),
        f"{other_label}_pre_stopped_by_convergence": bool(other_pre.get("stopped_by_convergence", False)),
        f"{base_label}_post_strategy": str(base_post.get("strategy", "")),
        f"{other_label}_post_strategy": str(other_post.get("strategy", "")),
        f"{base_label}_post_n_selected": int(base_post.get("n_selected", base["ablation_n_frames"])),
        f"{other_label}_post_n_selected": int(other_post.get("n_selected", other["ablation_n_frames"])),
        f"{base_label}_rejected_reason_counts": dict(base.get("rejected_reason_counts", {})),
        f"{other_label}_rejected_reason_counts": dict(other.get("rejected_reason_counts", {})),
        "same_final_basin_count": int(base["workflow_n_basins"]) == int(other["workflow_n_basins"]),
    }


def _build_summary(case_rows: list[dict[str, Any]], *, base_label: str, other_label: str) -> dict[str, Any]:
    if not case_rows:
        return {"n_cases": 0}
    n_cases = len(case_rows)
    base_pre = [int(r[f"{base_label}_pre_n_selected"]) for r in case_rows]
    other_pre = [int(r[f"{other_label}_pre_n_selected"]) for r in case_rows]
    base_post = [int(r[f"{base_label}_post_n_selected"]) for r in case_rows]
    other_post = [int(r[f"{other_label}_post_n_selected"]) for r in case_rows]
    base_basins = [int(r[f"{base_label}_workflow_n_basins"]) for r in case_rows]
    other_basins = [int(r[f"{other_label}_workflow_n_basins"]) for r in case_rows]
    return {
        "n_cases": int(n_cases),
        "n_same_final_basin_count": int(sum(bool(r["same_final_basin_count"]) for r in case_rows)),
        f"{base_label}_pre_selected_mean": float(sum(base_pre) / n_cases),
        f"{other_label}_pre_selected_mean": float(sum(other_pre) / n_cases),
        f"{base_label}_post_selected_mean": float(sum(base_post) / n_cases),
        f"{other_label}_post_selected_mean": float(sum(other_post) / n_cases),
        f"{base_label}_workflow_n_basins_mean": float(sum(base_basins) / n_cases),
        f"{other_label}_workflow_n_basins_mean": float(sum(other_basins) / n_cases),
        f"{base_label}_n_convergence_stops": int(sum(bool(r[f"{base_label}_pre_stopped_by_convergence"]) for r in case_rows)),
        f"{other_label}_n_convergence_stops": int(sum(bool(r[f"{other_label}_pre_stopped_by_convergence"]) for r in case_rows)),
    }


def _to_markdown(case_rows: list[dict[str, Any]], summary: dict[str, Any], *, base_label: str, other_label: str) -> str:
    lines = [
        "# Polyatomic Schedule Ablation",
        "",
        "## Aggregate",
        "",
        f"- n_cases: {summary['n_cases']}",
        f"- same_final_basin_count: {summary['n_same_final_basin_count']}/{summary['n_cases']}",
        f"- {base_label} mean pre-selected: {summary[f'{base_label}_pre_selected_mean']:.2f}",
        f"- {other_label} mean pre-selected: {summary[f'{other_label}_pre_selected_mean']:.2f}",
        f"- {base_label} mean post-selected: {summary[f'{base_label}_post_selected_mean']:.2f}",
        f"- {other_label} mean post-selected: {summary[f'{other_label}_post_selected_mean']:.2f}",
        f"- {base_label} convergence stops: {summary[f'{base_label}_n_convergence_stops']}",
        f"- {other_label} convergence stops: {summary[f'{other_label}_n_convergence_stops']}",
        "",
        "## Per Case",
        "",
        f"| case | {base_label} pre | {other_label} pre | {base_label} post | {other_label} post | {base_label} basins | {other_label} basins | same basins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['case']} | "
            f"{row[f'{base_label}_pre_n_selected']} | "
            f"{row[f'{other_label}_pre_n_selected']} | "
            f"{row[f'{base_label}_post_n_selected']} | "
            f"{row[f'{other_label}_post_n_selected']} | "
            f"{row[f'{base_label}_workflow_n_basins']} | "
            f"{row[f'{other_label}_workflow_n_basins']} | "
            f"{'yes' if row['same_final_basin_count'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=str, required=True)
    parser.add_argument("--other-summary", type=str, required=True)
    parser.add_argument("--base-label", type=str, default="default")
    parser.add_argument("--other-label", type=str, default="iterative_grid")
    parser.add_argument("--out-root", type=str, required=True)
    args = parser.parse_args()

    base_summary = Path(args.base_summary)
    other_summary = Path(args.other_summary)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_rows = _rows_by_case(base_summary)
    other_rows = _rows_by_case(other_summary)
    common_cases = sorted(set(base_rows).intersection(other_rows))
    case_rows = [
        _build_case_row(case, base_rows[case], other_rows[case], base_label=str(args.base_label), other_label=str(args.other_label))
        for case in common_cases
    ]
    summary = _build_summary(case_rows, base_label=str(args.base_label), other_label=str(args.other_label))
    payload = {
        "base_summary": base_summary.as_posix(),
        "other_summary": other_summary.as_posix(),
        "base_label": str(args.base_label),
        "other_label": str(args.other_label),
        "summary": summary,
        "rows": case_rows,
    }
    (out_root / "schedule_ablation_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_root / "schedule_ablation_comparison.md").write_text(
        _to_markdown(case_rows, summary, base_label=str(args.base_label), other_label=str(args.other_label)),
        encoding="utf-8",
    )
    print((out_root / "schedule_ablation_comparison.json").as_posix())
    print((out_root / "schedule_ablation_comparison.md").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
