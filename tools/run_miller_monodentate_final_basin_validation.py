from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.benchmark import audit_cu111_co_case, build_ase_reference_frames, select_unique_reference_matches
from adsorption_ensemble.basin.dedup import symmetry_aware_kabsch_rmsd
from adsorption_ensemble.pose import PoseSampler
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    DEFAULT_BASIN_DEDUP_METRIC,
    DEFAULT_MACE_NODE_L2_THRESHOLD,
    generate_adsorption_ensemble,
    list_sampling_schedule_presets,
    make_sampling_schedule,
)
from tools.run_miller_monodentate_matrix import build_miller_metal_slab_suite, build_monodentate_suite


def _make_relax_backend() -> MACEBatchRelaxBackend:
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path="/root/.cache/mace/mace-omat-0-small.model",
            device="cuda",
            dtype="float32",
            max_edges_per_batch=20000,
            head_name="omat_pbe",
            enable_cueq=True,
            strict=True,
        )
    )


def _manual_basin_config(work_dir: Path, *, dedup_metric: str, signature_mode: str, post_relax_selection: object | None) -> BasinConfig:
    return BasinConfig(
        relax_maxf=0.1,
        relax_steps=80,
        energy_window_ev=2.5,
        dedup_metric=str(dedup_metric),
        signature_mode=str(signature_mode),
        dedup_cluster_method="hierarchical",
        rmsd_threshold=0.10,
        mace_model_path="/root/.cache/mace/mace-omat-0-small.model",
        mace_device="cuda",
        mace_dtype="float32",
        mace_enable_cueq=True,
        mace_head_name="omat_pbe",
        mace_node_l2_threshold=DEFAULT_MACE_NODE_L2_THRESHOLD,
        final_basin_merge_metric="off",
        final_basin_merge_node_l2_threshold=DEFAULT_MACE_NODE_L2_THRESHOLD,
        final_basin_merge_cluster_method="hierarchical",
        desorption_min_bonds=1,
        post_relax_selection=post_relax_selection,
        work_dir=work_dir,
    )


def _manual_frames_for_ase_sites(slab: Atoms, adsorbate: Atoms) -> tuple[list[Atoms], list[dict[str, Any]]]:
    return build_ase_reference_frames(slab=slab, adsorbate=adsorbate)


def _reference_source_from_meta(meta: list[dict[str, Any]]) -> str:
    sources = sorted({str(row.get("reference_source", "")).strip() for row in meta if str(row.get("reference_source", "")).strip()})
    if not sources:
        return ""
    if len(sources) == 1:
        return str(sources[0])
    return "mixed"


def _load_basins(extxyz_path: str | None) -> list[Atoms]:
    if not extxyz_path:
        return []
    try:
        return list(read(extxyz_path, index=":"))
    except Exception:
        return []


def _overlap(manual_basins: list[Atoms], ours_basins: list[Atoms], slab_n: int, rmsd_threshold: float = 0.20) -> dict[str, Any]:
    candidates = []
    for j, b in enumerate(manual_basins):
        b_pos = np.asarray(b.get_positions(), dtype=float)[slab_n:]
        b_ads = b[slab_n:]
        for i, a in enumerate(ours_basins):
            sig_a = str(a.info.get("signature", ""))
            sig_b = str(b.info.get("signature", ""))
            a_pos = np.asarray(a.get_positions(), dtype=float)[slab_n:]
            d = float(symmetry_aware_kabsch_rmsd(a_pos, b_pos, b_ads))
            if not np.isfinite(d):
                continue
            if float(d) > float(rmsd_threshold):
                continue
            candidates.append(
                {
                    "manual_index": int(j),
                    "ours_index": int(i),
                    "signature": sig_a,
                    "signature_match": bool(sig_a == sig_b and sig_a.strip()),
                    "rmsd": float(d),
                }
            )
    matches = select_unique_reference_matches(
        candidates,
        n_manual=int(len(manual_basins)),
        n_ours=int(len(ours_basins)),
    )
    matched = int(len(matches))
    n_manual = int(len(manual_basins))
    n_ours = int(len(ours_basins))
    recall: float | None
    reference_state: str
    if n_manual <= 0 and n_ours <= 0:
        recall = None
        reference_state = "empty_agreement"
    elif n_manual <= 0:
        recall = None
        reference_state = "empty_reference_ours_positive"
    else:
        recall = float(matched / max(1, n_manual))
        reference_state = "manual_positive"
    return {
        "matched_manual_basins": int(matched),
        "n_manual_basins": int(n_manual),
        "n_ours_basins": int(n_ours),
        "n_unique_ours_matches": int(len({int(row["ours_index"]) for row in matches})),
        "manual_recall_by_ours": recall,
        "manual_reference_state": str(reference_state),
        "matches": matches,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_miller_metal_slab_suite()
    molecules = build_monodentate_suite()
    slab_names = sorted(slabs.keys())
    if args.slabs:
        allow = {v.strip() for v in str(args.slabs).split(",") if v.strip()}
        slab_names = [v for v in slab_names if v in allow]
    mol_names = sorted(molecules.keys())
    if args.molecules:
        allow = {v.strip() for v in str(args.molecules).split(",") if v.strip()}
        mol_names = [v for v in mol_names if v in allow]
    if int(args.max_slabs) > 0:
        slab_names = slab_names[: int(args.max_slabs)]
    if int(args.max_molecules) > 0:
        mol_names = mol_names[: int(args.max_molecules)]

    relax_backend = _make_relax_backend()
    schedule = make_sampling_schedule(str(args.schedule_preset))
    rows: list[dict[str, Any]] = []
    for slab_name in slab_names:
        slab = slabs[slab_name]
        for mol_name in mol_names:
            case_dir = out_root / slab_name / mol_name
            case_dir.mkdir(parents=True, exist_ok=True)
            ads = molecules[mol_name].copy()

            ours_result = generate_adsorption_ensemble(
                slab=slab,
                adsorbate=ads,
                work_dir=case_dir / "ours",
                placement_mode=str(args.placement_mode),
                schedule=schedule,
                dedup_metric=str(args.dedup_metric),
                signature_mode=str(args.signature_mode),
                basin_overrides={
                    "mace_model_path": "/root/.cache/mace/mace-omat-0-small.model",
                    "mace_device": "cuda",
                    "mace_dtype": "float32",
                    "mace_head_name": "omat_pbe",
                    "mace_node_l2_threshold": DEFAULT_MACE_NODE_L2_THRESHOLD,
                    "final_basin_merge_metric": "off",
                    "final_basin_merge_node_l2_threshold": DEFAULT_MACE_NODE_L2_THRESHOLD,
                    "final_basin_merge_cluster_method": "hierarchical",
                },
                basin_relax_backend=relax_backend,
            )
            ours_basins = _load_basins(ours_result.files.get("basins_extxyz"))

            manual_frames, manual_meta = _manual_frames_for_ase_sites(slab=slab, adsorbate=ads)
            manual_cfg = _manual_basin_config(
                case_dir / "ase_manual" / "basin_work",
                dedup_metric=str(args.dedup_metric),
                signature_mode=str(args.signature_mode),
                post_relax_selection=schedule.post_relax_selection,
            )
            manual_result = BasinBuilder(config=manual_cfg, relax_backend=relax_backend).build(
                frames=manual_frames,
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=2,
            )
            manual_basins: list[Atoms] = []
            for basin in manual_result.basins:
                a = basin.atoms.copy()
                a.info["signature"] = str(basin.signature)
                a.info["basin_id"] = int(basin.basin_id)
                a.info["energy_ev"] = float(basin.energy_ev)
                manual_basins.append(a)
            if manual_frames:
                write((case_dir / "ase_manual" / "manual_input_frames.extxyz").as_posix(), manual_frames)
            if manual_basins:
                write((case_dir / "ase_manual" / "manual_basins.extxyz").as_posix(), manual_basins)
            _write_json(
                case_dir / "ase_manual" / "manual_basin_summary.json",
                {
                    "reference_source": _reference_source_from_meta(manual_meta),
                    "manual_sites": manual_meta,
                    "binding_atom_index": int(manual_meta[0]["binding_atom_index"]) if manual_meta else None,
                    "binding_atom_symbol": (None if not manual_meta else str(manual_meta[0]["binding_atom_symbol"])),
                    "summary": manual_result.summary,
                    "rejected": [{"candidate_id": int(r.candidate_id), "reason": str(r.reason), "metrics": dict(r.metrics)} for r in manual_result.rejected],
                },
            )

            overlap = _overlap(manual_basins=manual_basins, ours_basins=ours_basins, slab_n=len(slab), rmsd_threshold=float(args.rmsd_threshold))
            rows.append(
                {
                    "case": f"{slab_name}__{mol_name}",
                    "slab": slab_name,
                    "adsorbate": mol_name,
                    "ours": {
                        "n_basis_primitives": int(ours_result.summary["n_basis_primitives"]),
                        "n_pose_frames": int(ours_result.summary["n_pose_frames"]),
                        "n_pose_frames_selected_for_basin": int(ours_result.summary["n_pose_frames_selected_for_basin"]),
                        "n_basins": int(ours_result.summary["n_basins"]),
                        "placement_mode": str(args.placement_mode),
                        "schedule_name": str(ours_result.summary["schedule"]["name"]),
                        "work_dir": (case_dir / "ours").as_posix(),
                    },
                    "ase_manual": {
                        "reference_source": _reference_source_from_meta(manual_meta),
                        "n_input_sites": int(len(manual_frames)),
                        "n_basins": int(len(manual_basins)),
                        "rejected_reason_counts": dict({str(k): int(v) for k, v in Counter(str(r.reason) for r in manual_result.rejected).items()}),
                        "work_dir": (case_dir / "ase_manual").as_posix(),
                    },
                    "overlap": overlap,
                }
            )
            if str(slab_name) == "Cu_fcc111" and str(mol_name) == "CO":
                sentinel = audit_cu111_co_case(case_dir)
                _write_json(case_dir / "cu111_co_sentinel_audit.json", sentinel)
                rows[-1]["sentinel_audit"] = sentinel
            _write_json(case_dir / "case_summary.json", rows[-1])

    recalls = [
        float(r["overlap"]["manual_recall_by_ours"])
        for r in rows
        if r["overlap"].get("manual_recall_by_ours", None) is not None
    ]
    total_pose_frames = int(sum(int(r["ours"]["n_pose_frames"]) for r in rows))
    total_pose_frames_selected_for_basin = int(sum(int(r["ours"]["n_pose_frames_selected_for_basin"]) for r in rows))
    total_ours_basins = int(sum(int(r["ours"]["n_basins"]) for r in rows))
    total_manual_basins = int(sum(int(r["ase_manual"]["n_basins"]) for r in rows))
    empty_reference_rows = [dict(r) for r in rows if int(r["ase_manual"]["n_basins"]) <= 0]
    empty_reference_agreement_rows = [
        dict(r)
        for r in rows
        if str(r["overlap"].get("manual_reference_state", "")) == "empty_agreement"
    ]
    empty_reference_ours_positive_rows = [
        dict(r)
        for r in rows
        if str(r["overlap"].get("manual_reference_state", "")) == "empty_reference_ours_positive"
    ]
    sentinel_rows = [dict(r) for r in rows if isinstance(r.get("sentinel_audit"), dict)]
    suspicious_rows = [
        dict(r)
        for r in sentinel_rows
        if str(r["sentinel_audit"].get("interpretation", "")) == "suspicious_hollow_collapse"
    ]
    payload = {
        "out_root": out_root.as_posix(),
        "mace_model_path": "/root/.cache/mace/mace-omat-0-small.model",
        "dedup_metric": str(args.dedup_metric),
        "signature_mode": str(args.signature_mode),
        "placement_mode": str(args.placement_mode),
        "schedule_preset": str(args.schedule_preset),
        "schedule_name": str(schedule.name),
        "n_cases": int(len(rows)),
        "total_pose_frames": int(total_pose_frames),
        "total_pose_frames_selected_for_basin": int(total_pose_frames_selected_for_basin),
        "total_ours_basins": int(total_ours_basins),
        "total_manual_basins": int(total_manual_basins),
        "n_recall_defined_cases": int(len(recalls)),
        "mean_manual_recall_by_ours": (None if not recalls else float(np.mean(recalls))),
        "min_manual_recall_by_ours": (None if not recalls else float(np.min(recalls))),
        "n_full_recall": int(sum(1 for v in recalls if abs(v - 1.0) <= 1e-12)),
        "n_empty_reference_cases": int(len(empty_reference_rows)),
        "n_empty_reference_agreement_cases": int(len(empty_reference_agreement_rows)),
        "n_empty_reference_ours_positive_cases": int(len(empty_reference_ours_positive_rows)),
        "n_sentinel_cases": int(len(sentinel_rows)),
        "n_suspicious_hollow_collapse_cases": int(len(suspicious_rows)),
        "suspicious_hollow_collapse_cases": [
            {
                "case": str(row["case"]),
                "work_dir": str(row["ours"]["work_dir"]),
                "interpretation": str(row["sentinel_audit"].get("interpretation", "")),
                "coordination": int(row["sentinel_audit"]["final_binding_environment"].get("coordination", -1)),
                "n_basis_sites": int(row["sentinel_audit"].get("n_basis_sites", 0)),
                "workflow_n_basins": int(row["sentinel_audit"].get("workflow_n_basins", 0)),
            }
            for row in suspicious_rows
        ],
        "rows": rows,
    }
    out_path = out_root / "miller_monodentate_final_basin_validation.json"
    _write_json(out_path, payload)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/final_basin_validation/miller_monodentate")
    parser.add_argument("--slabs", type=str, default="")
    parser.add_argument("--molecules", type=str, default="H,CO,H2O,NH3")
    parser.add_argument("--max-slabs", type=int, default=0)
    parser.add_argument("--max-molecules", type=int, default=0)
    parser.add_argument("--rmsd-threshold", type=float, default=0.20)
    parser.add_argument("--dedup-metric", type=str, default=DEFAULT_BASIN_DEDUP_METRIC)
    parser.add_argument("--signature-mode", type=str, default="provenance", choices=["absolute", "canonical", "provenance", "none"])
    parser.add_argument("--placement-mode", type=str, default="anchor_free", choices=["anchor_free", "anchor_aware"])
    parser.add_argument("--schedule-preset", type=str, default="multistage_default", choices=list_sampling_schedule_presets())
    args = parser.parse_args()
    out_path = run(args)
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
