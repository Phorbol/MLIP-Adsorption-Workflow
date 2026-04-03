from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ase.io import read

from adsorption_ensemble.basin import BasinConfig, run_named_basin_ablation
from adsorption_ensemble.basin.dedup import build_binding_pairs, binding_signature
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.selection import StageSelectionConfig
from adsorption_ensemble.workflows import SamplingSchedule, evaluate_adsorption_workflow_readiness, generate_adsorption_ensemble
from tools.run_autoresearch_artifact_suite import build_slab_cases, infer_mace_head_name, resolve_mace_model_path, runtime_manifest
from tools.run_polyatomic_final_dedup_suite import DEFAULT_CASES, _parse_cases, _write_json
from tests.chemistry_cases import get_test_adsorbate_cases


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("budget list is empty")
    return out


def _build_relax_backend(*, model_path: str | None, device: str, dtype: str) -> MACEBatchRelaxBackend:
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=str(device),
            dtype=str(dtype),
            max_edges_per_batch=20000,
            head_name=infer_mace_head_name(model_path),
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
    }
    return {
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
        "pure_rmsd": BasinConfig(
            dedup_metric="pure_rmsd",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "pure_mace_0p05": BasinConfig(
            dedup_metric="pure_mace",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.05,
            mace_model_path=model_path,
            mace_device=str(device),
            mace_dtype=str(dtype),
            **common,
        ),
    }


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


def _load_basin_signatures(case_dir: Path) -> list[str]:
    basins_json = case_dir / "basins.json"
    if not basins_json.exists():
        return []
    try:
        payload = json.loads(basins_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    return sorted(str(x.get("signature", "")) for x in payload.get("basins", []))


def _load_binding_pair_sets(case_dir: Path, *, slab_n: int, binding_tau: float = 1.15) -> list[tuple[tuple[int, int], ...]]:
    basins_extxyz = case_dir / "basins.extxyz"
    if not basins_extxyz.exists():
        return []
    try:
        frames = list(read(basins_extxyz.as_posix(), index=":"))
    except Exception:
        return []
    motifs: list[tuple[tuple[int, int], ...]] = []
    for atoms in frames:
        pairs = build_binding_pairs(atoms, slab_n=int(slab_n), binding_tau=float(binding_tau))
        motifs.append(tuple((int(i), int(j)) for i, j in pairs))
    return sorted(motifs)


def _load_canonical_signature_set(case_dir: Path, *, slab_n: int, binding_tau: float = 1.15) -> list[str]:
    basins_extxyz = case_dir / "basins.extxyz"
    if not basins_extxyz.exists():
        return []
    try:
        frames = list(read(basins_extxyz.as_posix(), index=":"))
    except Exception:
        return []
    signatures = []
    for atoms in frames:
        pairs = build_binding_pairs(atoms, slab_n=int(slab_n), binding_tau=float(binding_tau))
        signatures.append(binding_signature(pairs, frame=atoms, slab_n=int(slab_n), mode="canonical"))
    return sorted(str(x) for x in signatures)


def _enrich_row_with_baseline_matches(row: dict[str, Any], *, baseline_dir: Path | None, slab_n: int) -> dict[str, Any]:
    payload = dict(row)
    prov_match = payload.get("matches_baseline_signature_set", None)
    payload["matches_baseline_provenance_signature_set"] = prov_match
    if baseline_dir is None or not baseline_dir.exists():
        payload["matches_baseline_binding_pairs_set"] = None
        payload["matches_baseline_canonical_signature_set"] = None
        return payload
    case_dir = Path(str(payload["work_dir"]))
    payload["matches_baseline_binding_pairs_set"] = (
        _load_binding_pair_sets(case_dir, slab_n=int(slab_n)) == _load_binding_pair_sets(baseline_dir, slab_n=int(slab_n))
    )
    payload["matches_baseline_canonical_signature_set"] = (
        _load_canonical_signature_set(case_dir, slab_n=int(slab_n)) == _load_canonical_signature_set(baseline_dir, slab_n=int(slab_n))
    )
    return payload


def _make_schedule(max_candidates: int) -> SamplingSchedule:
    return SamplingSchedule(
        name=f"fps_budget_{int(max_candidates)}",
        exhaustive_pose_sampling=False,
        run_conformer_search=False,
        pre_relax_selection=StageSelectionConfig(
            enabled=True,
            strategy="fps",
            max_candidates=int(max_candidates),
            descriptor="adsorbate_surface_distance",
            random_seed=0,
        ),
        post_relax_selection=StageSelectionConfig(
            enabled=True,
            strategy="energy_rmsd_window",
            energy_window_ev=3.0,
            rmsd_threshold=0.05,
            descriptor="adsorbate_surface_distance",
        ),
        notes="Fixed-budget FPS budget sweep with the current controlled workflow settings.",
    )


def _rows_to_markdown(rows: list[dict[str, Any]], *, budgets: list[int]) -> str:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["budget_k"])].append(row)
    lines = [
        "# Polyatomic Pre-Relax Budget Sweep",
        "",
        "## Aggregate",
        "",
        "| budget_k | n_cases | exact_basin_count_match | exact_provenance_signature_match | exact_binding_pairs_match | exact_canonical_signature_match | mean_selected_for_basin | mean_workflow_basins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in budgets:
        items = grouped.get(int(budget), [])
        if not items:
            continue
        n = len(items)
        prov_total = sum(x.get("matches_baseline_provenance_signature_set") is not None for x in items)
        pair_total = sum(x.get("matches_baseline_binding_pairs_set") is not None for x in items)
        canonical_total = sum(x.get("matches_baseline_canonical_signature_set") is not None for x in items)
        lines.append(
            f"| {int(budget)} | {n} | "
            f"{sum(bool(x['matches_baseline_basin_count']) for x in items)}/{n} | "
            f"{sum(bool(x.get('matches_baseline_provenance_signature_set')) for x in items)}/{prov_total if prov_total > 0 else n} | "
            f"{sum(bool(x.get('matches_baseline_binding_pairs_set')) for x in items)}/{pair_total if pair_total > 0 else n} | "
            f"{sum(bool(x.get('matches_baseline_canonical_signature_set')) for x in items)}/{canonical_total if canonical_total > 0 else n} | "
            f"{sum(int(x['n_pose_frames_selected_for_basin']) for x in items)/n:.2f} | "
            f"{sum(int(x['workflow_n_basins']) for x in items)/n:.2f} |"
        )
    lines += [
        "",
        "## Per Case",
        "",
        "| case | budget_k | selected_for_basin | workflow_basins | baseline_basins | count_match | provenance_match | binding_pairs_match | canonical_match |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        prov = row.get("matches_baseline_provenance_signature_set", None)
        pair = row.get("matches_baseline_binding_pairs_set", None)
        canonical = row.get("matches_baseline_canonical_signature_set", None)
        lines.append(
            f"| {row['case']} | {row['budget_k']} | {row['n_pose_frames_selected_for_basin']} | "
            f"{row['workflow_n_basins']} | {row['baseline_workflow_n_basins']} | "
            f"{'yes' if row['matches_baseline_basin_count'] else 'no'} | "
            f"{'n/a' if prov is None else ('yes' if prov else 'no')} | "
            f"{'n/a' if pair is None else ('yes' if pair else 'no')} | "
            f"{'n/a' if canonical is None else ('yes' if canonical else 'no')} |"
        )
    return "\n".join(lines) + "\n"


def _collect_existing_rows(out_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_root.glob("k_*/*/*/budget_sweep_summary.json")):
        try:
            rows.append(dict(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return rows


def _write_summary_outputs(*, out_root: Path, rows: list[dict[str, Any]], budgets: list[int] | None = None) -> Path:
    if not rows:
        raise ValueError("No budget sweep rows available to summarize.")
    summary_json = out_root / "polyatomic_pre_relax_budget_sweep_summary.json"
    summary_csv = out_root / "polyatomic_pre_relax_budget_sweep_summary.csv"
    summary_md = out_root / "polyatomic_pre_relax_budget_sweep_summary.md"
    rows_sorted = sorted(rows, key=lambda x: (int(x.get("budget_k", 0)), str(x.get("case", ""))))
    used_budgets = sorted(set(int(x.get("budget_k", 0)) for x in rows_sorted)) if budgets is None else sorted(set(int(x) for x in budgets))
    _write_json(summary_json, rows_sorted)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows_sorted[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    summary_md.write_text(_rows_to_markdown(rows_sorted, budgets=used_budgets), encoding="utf-8")
    print(summary_json.as_posix())
    print(summary_csv.as_posix())
    print(summary_md.as_posix())
    return summary_json


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_slab_cases()
    adsorbates = get_test_adsorbate_cases()
    cases = _parse_cases(str(args.cases)) if str(args.cases).strip() else _parse_cases(",".join(DEFAULT_CASES))
    budgets = sorted(set(_parse_int_list(str(args.budgets))))

    baseline_rows = {}
    if str(args.baseline_summary).strip():
        baseline_rows = {
            str(row["case"]): dict(row)
            for row in json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        }

    if bool(args.summarize_only):
        rows = []
        for row in _collect_existing_rows(out_root):
            baseline = baseline_rows.get(str(row.get("case", "")), {})
            baseline_dir = (None if not baseline else Path(str(baseline["work_dir"])))
            slab_name = str(row["slab"])
            row_enriched = _enrich_row_with_baseline_matches(row, baseline_dir=baseline_dir, slab_n=len(slabs[slab_name]))
            rows.append(row_enriched)
        return _write_summary_outputs(out_root=out_root, rows=rows, budgets=budgets)

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
    ablation_cfgs = _default_ablation_configs(model_path=model_path, device=str(args.mace_device), dtype=str(args.mace_dtype))

    rows = []
    for budget_k in budgets:
        schedule = _make_schedule(int(budget_k))
        for slab_name, ads_name in cases:
            case = f"{slab_name}__{ads_name}"
            case_dir = out_root / f"k_{int(budget_k)}" / slab_name / ads_name
            slab = slabs[slab_name]
            adsorbate = adsorbates[ads_name]
            result = generate_adsorption_ensemble(
                slab=slab,
                adsorbate=adsorbate,
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
                    "desorption_min_bonds": 1,
                    "energy_window_ev": 2.5,
                },
                basin_relax_backend=relax_backend,
            )
            readiness = evaluate_adsorption_workflow_readiness(result.workflow)
            ablation_input_path = case_dir / "basin_work" / "post_relax_selected.extxyz"
            if not ablation_input_path.exists():
                ablation_input_path = case_dir / "basin_work" / "relax" / "relaxed_stream.extxyz"
            relaxed_frames = list(read(ablation_input_path.as_posix(), index=":"))
            ablation = run_named_basin_ablation(
                frames=relaxed_frames,
                slab_ref=slab,
                adsorbate_ref=adsorbate,
                slab_n=len(slab),
                normal_axis=2,
                configs=ablation_cfgs,
                relax_backend=None,
            )
            _write_json(case_dir / "budget_sweep_ablation.json", ablation)
            baseline = baseline_rows.get(case, {})
            basin_signatures = _load_basin_signatures(case_dir)
            baseline_signatures = []
            if baseline:
                baseline_signatures = _load_basin_signatures(Path(str(baseline["work_dir"])))
            row = {
                "budget_k": int(budget_k),
                "case": case,
                "slab": str(slab_name),
                "adsorbate": str(ads_name),
                "work_dir": case_dir.as_posix(),
                "n_pose_frames": int(result.summary["n_pose_frames"]),
                "n_pose_frames_selected_for_basin": int(result.summary["n_pose_frames_selected_for_basin"]),
                "workflow_n_basins": int(result.summary["n_basins"]),
                "paper_readiness_score": int(readiness.score),
                "paper_readiness_max_score": int(readiness.max_score),
                "ablation_n_frames": int(len(relaxed_frames)),
                "binding_surface_greedy_n_basins": int(ablation["configs"]["binding_surface_greedy"]["n_basins"]),
                "pure_rmsd_n_basins": int(ablation["configs"]["pure_rmsd"]["n_basins"]),
                "pure_mace_0p05_n_basins": int(ablation["configs"]["pure_mace_0p05"]["n_basins"]),
                "rejected_reason_counts": _load_rejected_reason_counts(case_dir),
                "basin_signatures": list(basin_signatures),
                "baseline_workflow_n_basins": (None if not baseline else int(baseline["workflow_n_basins"])),
                "matches_baseline_basin_count": (None if not baseline else int(result.summary["n_basins"]) == int(baseline["workflow_n_basins"])),
                "matches_baseline_signature_set": (None if not baseline_signatures else list(basin_signatures) == list(baseline_signatures)),
            }
            row = _enrich_row_with_baseline_matches(
                row,
                baseline_dir=(None if not baseline else Path(str(baseline["work_dir"]))),
                slab_n=len(slab),
            )
            _write_json(case_dir / "budget_sweep_summary.json", row)
            rows.append(row)

    return _write_summary_outputs(out_root=out_root, rows=rows, budgets=budgets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/polyatomic_pre_relax_budget_sweep")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--budgets", type=str, default="8,12,16,24")
    parser.add_argument("--baseline-summary", type=str, default="")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--summarize-only", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
