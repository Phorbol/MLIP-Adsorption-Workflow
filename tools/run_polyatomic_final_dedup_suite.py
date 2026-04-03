from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read

from adsorption_ensemble.basin import BasinConfig, run_named_basin_ablation
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import evaluate_adsorption_workflow_readiness, generate_adsorption_ensemble, make_sampling_schedule
from tools.run_autoresearch_artifact_suite import build_slab_cases, infer_mace_head_name, resolve_mace_model_path, runtime_manifest
from tests.chemistry_cases import get_test_adsorbate_cases


DEFAULT_CASES = (
    "fcc111__NH3",
    "fcc111__H2O",
    "fcc111__CH3OH",
    "fcc111__C2H4",
    "fcc111__C6H6",
    "fcc211__CH3OH",
    "cu321__C6H6",
    "cu321__p_nitrochlorobenzene_like",
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
        "prov_rmsd": BasinConfig(
            dedup_metric="rmsd",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "pure_rmsd": BasinConfig(
            dedup_metric="pure_rmsd",
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
        "binding_surface_hierarchical": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
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


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_slab_cases()
    adsorbates = get_test_adsorbate_cases()
    cases = _parse_cases(str(args.cases)) if str(args.cases).strip() else _parse_cases(",".join(DEFAULT_CASES))

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

    rows = []
    for slab_name, ads_name in cases:
        if slab_name not in slabs:
            raise KeyError(f"Unknown slab case: {slab_name}")
        if ads_name not in adsorbates:
            raise KeyError(f"Unknown adsorbate case: {ads_name}")
        case_dir = out_root / slab_name / ads_name
        if case_dir.exists() and not bool(args.rerun_existing):
            summary_path = case_dir / "final_dedup_suite_summary.json"
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue

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
        _write_json(case_dir / "final_dedup_suite_ablation.json", ablation)
        rejected_reason_counts = _load_rejected_reason_counts(case_dir)
        row = {
            "slab": str(slab_name),
            "adsorbate": str(ads_name),
            "case": f"{slab_name}__{ads_name}",
            "work_dir": case_dir.as_posix(),
            "ablation_input_path": ablation_input_path.as_posix(),
            "ablation_n_frames": int(len(relaxed_frames)),
            "n_surface_atoms": int(result.summary["n_surface_atoms"]),
            "n_basis_primitives": int(result.summary["n_basis_primitives"]),
            "n_pose_frames": int(result.summary["n_pose_frames"]),
            "n_pose_frames_selected_for_basin": int(result.summary.get("n_pose_frames_selected_for_basin", result.summary["n_pose_frames"])),
            "workflow_n_basins": int(result.summary["n_basins"]),
            "paper_readiness_score": int(readiness.score),
            "paper_readiness_max_score": int(readiness.max_score),
            "rejected_reason_counts": rejected_reason_counts,
            "prov_rmsd_n_basins": int(ablation["configs"]["prov_rmsd"]["n_basins"]),
            "pure_rmsd_n_basins": int(ablation["configs"]["pure_rmsd"]["n_basins"]),
            "binding_surface_greedy_n_basins": int(ablation["configs"]["binding_surface_greedy"]["n_basins"]),
            "binding_surface_hierarchical_n_basins": int(ablation["configs"]["binding_surface_hierarchical"]["n_basins"]),
            "pure_mace_0p05_n_basins": int(ablation["configs"]["pure_mace_0p05"]["n_basins"]),
        }
        _write_json(case_dir / "final_dedup_suite_summary.json", row)
        rows.append(row)

    summary_json = out_root / "polyatomic_final_dedup_suite_summary.json"
    summary_csv = out_root / "polyatomic_final_dedup_suite_summary.csv"
    _write_json(summary_json, rows)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    print(summary_json.as_posix())
    print(summary_csv.as_posix())
    return summary_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/polyatomic_final_dedup_suite")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--schedule-preset", type=str, default="multistage_default")
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--rerun-existing", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
