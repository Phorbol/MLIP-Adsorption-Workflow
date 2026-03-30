from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from tools.run_autoresearch_artifact_suite import (
    build_config,
    build_slab_cases,
    infer_mace_head_name,
    resolve_mace_model_path,
    runtime_manifest,
)
from adsorption_ensemble.workflows import evaluate_adsorption_workflow_readiness, run_adsorption_workflow
from tests.chemistry_cases import get_test_adsorbate_cases


REQUIRED_CASE_FILES = (
    "sites.png",
    "sites_only.png",
    "sites_inequivalent.png",
    "site_embedding_pca.png",
    "raw_site_dictionary.json",
    "selected_site_dictionary.json",
    "basin_dictionary.json",
    "basin_ablation.json",
    "basins.extxyz",
    "basins.json",
    "nodes.json",
    "workflow_summary.json",
)


def _case_complete(case_dir: Path) -> tuple[bool, str]:
    for name in REQUIRED_CASE_FILES:
        if not (case_dir / name).exists():
            return False, f"missing_file:{name}"
    ab_path = case_dir / "basin_ablation.json"
    try:
        payload = json.loads(ab_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"bad_basin_ablation_json:{type(exc).__name__}"
    metrics = payload.get("metrics", {})
    for metric in ("signature_only", "rmsd", "mace_node_l2"):
        rec = metrics.get(metric, {})
        if str(rec.get("status", "")) != "ok":
            return False, f"ablation_not_ok:{metric}"
    return True, "ok"


def _discover_cases(root: Path) -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    for group in ("workflow_matrix", "real_cases"):
        base = root / group
        if not base.exists():
            continue
        for slab_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            for ads_dir in sorted([p for p in slab_dir.iterdir() if p.is_dir()]):
                cases.append((group, slab_dir.name, ads_dir.name))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=str, default="artifacts/autoresearch/review_ready_cases")
    parser.add_argument("--target-root", type=str, default="artifacts/autoresearch/review_ready_cases_omat_cuda_fp32")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--max-selected-primitives", type=int, default=24)
    parser.add_argument("--rerun-all", action="store_true")
    args = parser.parse_args()

    src = Path(args.source_root)
    dst = Path(args.target_root)
    dst.mkdir(parents=True, exist_ok=True)

    model_path, model_source = resolve_mace_model_path(args.mace_model_path)
    device_requested = str(args.mace_device)
    device_effective = str(device_requested)
    if str(device_requested).lower().startswith("cuda"):
        ok = False
        try:
            import torch  # type: ignore

            ok = bool(torch.cuda.is_available())
        except Exception:
            ok = False
        if not ok and bool(args.require_cuda):
            raise RuntimeError("CUDA is required for MACE, but torch.cuda.is_available() is False.")
    runtime_manifest(
        mace_model_path=model_path,
        model_source=model_source,
        mace_device=device_requested,
        mace_device_effective=device_effective,
        mace_dtype=str(args.mace_dtype),
        out_root=dst,
    )
    head_name = infer_mace_head_name(model_path)
    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=model_path,
            device=device_effective,
            dtype=str(args.mace_dtype),
            max_edges_per_batch=15000,
            head_name=head_name,
            strict=True,
        )
    )

    slabs = build_slab_cases()
    adsorbates = get_test_adsorbate_cases()

    discovered = _discover_cases(src)
    rows = []
    n_rerun = 0
    n_skip = 0
    for group, slab_name, ads_name in discovered:
        if slab_name not in slabs or ads_name not in adsorbates:
            rows.append(
                {
                    "group": group,
                    "slab": slab_name,
                    "adsorbate": ads_name,
                    "action": "skip_unknown_case",
                }
            )
            n_skip += 1
            continue
        src_case = src / group / slab_name / ads_name
        dst_case = dst / group / slab_name / ads_name
        complete, reason = _case_complete(src_case)
        if complete and not bool(args.rerun_all):
            rows.append(
                {
                    "group": group,
                    "slab": slab_name,
                    "adsorbate": ads_name,
                    "action": "copy_complete",
                    "source_check": reason,
                    "work_dir": src_case.as_posix(),
                }
            )
            n_skip += 1
            continue
        cfg = build_config(
            dst_case,
            mace_model_path=model_path,
            mace_device=device_effective,
            mace_dtype=str(args.mace_dtype),
            max_selected_primitives=int(args.max_selected_primitives),
        )
        result = run_adsorption_workflow(
            slab=slabs[slab_name],
            adsorbate=adsorbates[ads_name],
            config=cfg,
            basin_relax_backend=relax_backend,
        )
        ready = evaluate_adsorption_workflow_readiness(result)
        rows.append(
            {
                "group": group,
                "slab": slab_name,
                "adsorbate": ads_name,
                "action": "rerun",
                "source_check": reason,
                "n_surface_atoms": int(result.summary["n_surface_atoms"]),
                "n_raw_primitives": int(result.summary["n_raw_primitives"]),
                "n_selected_primitives": int(result.summary["n_primitives"]),
                "n_pose_frames": int(result.summary["n_pose_frames"]),
                "n_basins": int(result.summary["n_basins"]),
                "n_nodes": int(result.summary["n_nodes"]),
                "paper_readiness_score": int(ready.score),
                "paper_readiness_max_score": int(ready.max_score),
                "work_dir": dst_case.as_posix(),
            }
        )
        n_rerun += 1

    out = {
        "source_root": src.as_posix(),
        "target_root": dst.as_posix(),
        "mace_model_path": model_path,
        "mace_model_source": model_source,
        "mace_device_requested": device_requested,
        "mace_device_effective": device_effective,
        "mace_dtype": str(args.mace_dtype),
        "n_discovered_cases": len(discovered),
        "n_rerun": int(n_rerun),
        "n_skip": int(n_skip),
        "rows": rows,
    }
    (dst / "rerun_manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
