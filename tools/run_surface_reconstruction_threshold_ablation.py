from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ase.io import read

from adsorption_ensemble.basin import BasinConfig, run_named_basin_ablation
from tests.chemistry_cases import get_test_adsorbate_cases
from tools.run_ase_autoadsorbate_crosscheck import build_slab_suite


DEFAULT_CASES = (
    "TiO2_110__NH3",
    "MgO_100__NH3",
    "Pt_fcc111_adatom__CO",
    "Pt_fcc111_cluster_interface__C6H6",
)

DEFAULT_THRESHOLDS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_cases(raw: str) -> list[tuple[str, str]]:
    out = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        slab_name, ads_name = token.split("__", 1)
        out.append((str(slab_name), str(ads_name)))
    return out


def _parse_thresholds(raw: str) -> list[float]:
    vals = [float(x) for x in str(raw).split(",") if str(x).strip()]
    return vals or list(DEFAULT_THRESHOLDS)


def _case_dir_from_root(root: Path, slab_name: str, ads_name: str) -> Path:
    return root / slab_name / ads_name


def _load_relaxed_frames(case_dir: Path) -> list:
    candidates = (
        case_dir / "basin_work" / "post_relax_selected.extxyz",
        case_dir / "basin_work" / "relax" / "relaxed_stream.extxyz",
    )
    for path in candidates:
        if path.exists():
            return list(read(path.as_posix(), index=":"))
    raise FileNotFoundError(f"No relaxed frames found under {case_dir}")


def _load_basins_json(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "basins.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _make_config(
    *,
    work_dir: Path,
    threshold: float,
    mace_model_path: str,
    mace_device: str,
    mace_dtype: str,
    mace_head_name: str | None,
) -> BasinConfig:
    return BasinConfig(
        relax_maxf=0.1,
        relax_steps=80,
        energy_window_ev=2.5,
        dedup_metric="binding_surface_distance",
        signature_mode="provenance",
        dedup_cluster_method="greedy",
        surface_descriptor_threshold=0.30,
        surface_descriptor_nearest_k=8,
        surface_descriptor_atom_mode="binding_only",
        surface_descriptor_relative=False,
        surface_descriptor_rmsd_gate=0.25,
        final_basin_merge_metric="mace_node_l2",
        final_basin_merge_node_l2_threshold=0.20,
        final_basin_merge_cluster_method="hierarchical",
        desorption_min_bonds=1,
        surface_reconstruction_max_disp=float(threshold),
        mace_model_path=str(mace_model_path),
        mace_device=str(mace_device),
        mace_dtype=str(mace_dtype),
        mace_head_name=(None if mace_head_name is None else str(mace_head_name)),
        work_dir=work_dir,
    )


def run(args: argparse.Namespace) -> Path:
    suite_root = Path(args.suite_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_slab_suite()
    adsorbates = get_test_adsorbate_cases()
    cases = _parse_cases(str(args.cases)) if str(args.cases).strip() else _parse_cases(",".join(DEFAULT_CASES))
    thresholds = _parse_thresholds(str(args.thresholds))

    rows: list[dict[str, Any]] = []
    for slab_name, ads_name in cases:
        case_dir = _case_dir_from_root(suite_root, slab_name, ads_name)
        frames = _load_relaxed_frames(case_dir)
        slab = slabs[slab_name]
        adsorbate = adsorbates[ads_name]
        original = _load_basins_json(case_dir)
        configs = {
            f"thr_{threshold:.2f}".replace(".", "p"): _make_config(
                work_dir=out_root / slab_name / ads_name / f"thr_{threshold:.2f}".replace(".", "p"),
                threshold=float(threshold),
                mace_model_path=str(args.mace_model_path),
                mace_device=str(args.mace_device),
                mace_dtype=str(args.mace_dtype),
                mace_head_name=(None if not str(args.mace_head_name).strip() else str(args.mace_head_name)),
            )
            for threshold in thresholds
        }
        ablation = run_named_basin_ablation(
            frames=frames,
            slab_ref=slab,
            adsorbate_ref=adsorbate,
            slab_n=len(slab),
            normal_axis=2,
            configs=configs,
            relax_backend=None,
        )
        _write_json(out_root / slab_name / ads_name / "surface_reconstruction_threshold_ablation.json", ablation)
        row = {
            "case": f"{slab_name}__{ads_name}",
            "source_case_dir": case_dir.as_posix(),
            "original_summary": original.get("summary", {}),
            "original_rejected_reason_counts": {
                str(k): int(v)
                for k, v in _count_rejected_reasons(original).items()
            },
            "threshold_rows": [],
        }
        for name, result in ablation.get("configs", {}).items():
            row["threshold_rows"].append(
                {
                    "config": str(name),
                    "status": str(result.get("status", "")),
                    "n_basins": int(result.get("n_basins", 0)),
                    "n_rejected": int(result.get("n_rejected", 0)),
                    "rejected_reason_counts": {
                        str(k): int(v)
                        for k, v in _count_rejected_reasons(result.get("basin_dictionary", {})).items()
                    },
                    "summary": dict(result.get("summary", {})),
                }
            )
        rows.append(row)
        _write_json(out_root / slab_name / ads_name / "surface_reconstruction_threshold_case_summary.json", row)

    payload = {
        "suite_root": suite_root.as_posix(),
        "out_root": out_root.as_posix(),
        "cases": [f"{slab}__{ads}" for slab, ads in cases],
        "thresholds": [float(x) for x in thresholds],
        "mace_model_path": str(args.mace_model_path),
        "mace_device": str(args.mace_device),
        "mace_dtype": str(args.mace_dtype),
        "mace_head_name": (None if not str(args.mace_head_name).strip() else str(args.mace_head_name)),
        "rows": rows,
    }
    _write_json(out_root / "surface_reconstruction_threshold_ablation_summary.json", payload)
    md_lines = [
        "# Surface Reconstruction Threshold Ablation",
        "",
        f"- Cases: {len(rows)}",
        f"- Thresholds: {[float(x) for x in thresholds]}",
        "",
    ]
    for row in rows:
        md_lines.append(f"## {row['case']}")
        md_lines.append("")
        md_lines.append(f"- original_rejected_reason_counts: {row['original_rejected_reason_counts']}")
        for thr in row["threshold_rows"]:
            md_lines.append(
                f"- {thr['config']}: n_basins={thr['n_basins']}, n_rejected={thr['n_rejected']}, "
                f"rejected_reason_counts={thr['rejected_reason_counts']}"
            )
        md_lines.append("")
    (out_root / "surface_reconstruction_threshold_ablation_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )
    print((out_root / "surface_reconstruction_threshold_ablation_summary.json").as_posix())
    print((out_root / "surface_reconstruction_threshold_ablation_summary.md").as_posix())
    return out_root / "surface_reconstruction_threshold_ablation_summary.json"


def _count_rejected_reasons(payload: dict[str, Any]) -> dict[str, int]:
    rejected = payload.get("rejected", [])
    counts: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("reason", "")).strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-root",
        type=str,
        default="artifacts/autoresearch/heterogeneous_final_workflow_suite_20260405",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="artifacts/autoresearch/threshold_ablation/surface_reconstruction_20260405",
    )
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--thresholds", type=str, default="0.50,0.75,1.00,1.25,1.50,2.00")
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--mace-head-name", type=str, default="omat_pbe")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
