from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc111, fcc211
from ase.io import read, write

from adsorption_ensemble.basin.anomaly import classify_anomaly
from adsorption_ensemble.basin.dedup import build_adsorbate_bonds, build_binding_pairs
from autoadsorbate.Smile import atoms_from_smile
from tools.run_miller_monodentate_matrix import build_miller_metal_slab_suite, build_monodentate_suite


def _crosslib_case_registry() -> dict[str, dict[str, Any]]:
    return {
        "Pt111_methanol": {"slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CO")},
        "Pt100_methanol": {"slab": fcc100("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CO")},
        "Pt111_dimethyl_ether": {"slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("COC")},
        "Pt211_ethanol": {"slab": fcc211("Pt", size=(6, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CCO")},
        "Pt111_methylamine": {"slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CN")},
    }


def _miller_case_registry() -> dict[str, dict[str, Any]]:
    slabs = build_miller_metal_slab_suite()
    molecules = build_monodentate_suite()
    out: dict[str, dict[str, Any]] = {}
    for slab_name, slab in slabs.items():
        for mol_name, adsorbate in molecules.items():
            out[f"{slab_name}__{mol_name}"] = {"slab": slab, "adsorbate": adsorbate}
    return out


def _case_root(benchmark_root: Path, suite: str, case_name: str) -> Path:
    if suite == "crosslib":
        return benchmark_root / case_name / "ours"
    slab_name, ads_name = case_name.split("__", 1)
    return benchmark_root / slab_name / ads_name / "ours"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _focus_payload(frame: Atoms, *, idx: int, slab_n: int, adsorbate_ref: Atoms) -> dict[str, Any]:
    ads = frame[int(slab_n) :].copy()
    ads_pos = np.asarray(ads.get_positions(), dtype=float)
    slab_pos = np.asarray(frame.get_positions(), dtype=float)[: int(slab_n)]
    pairs = build_binding_pairs(frame, slab_n=int(slab_n), binding_tau=1.15)
    bonds_ref = build_adsorbate_bonds(adsorbate_ref.copy())
    bonds_now = build_adsorbate_bonds(ads.copy())
    pairwise = []
    syms = ads.get_chemical_symbols()
    for i in range(len(ads)):
        d_surf = np.sort(np.linalg.norm(slab_pos - ads_pos[i][None, :], axis=1))[:6]
        pairwise.append(
            {
                "ads_index": int(i),
                "symbol": str(syms[i]),
                "nearest_surface_distances": [float(x) for x in d_surf.tolist()],
            }
        )
    bond_lengths = []
    for i in range(len(ads)):
        for j in range(i + 1, len(ads)):
            bond_lengths.append(
                {
                    "pair": [int(i), int(j)],
                    "symbols": [str(syms[i]), str(syms[j])],
                    "distance": float(np.linalg.norm(ads_pos[i] - ads_pos[j])),
                }
            )
    return {
        "candidate_id": int(idx),
        "site_label": str(frame.info.get("site_label", "")),
        "basis_id": frame.info.get("basis_id", None),
        "binding_pairs": [(int(i), int(j)) for i, j in pairs],
        "adsorbate_ref_bonds": [(int(i), int(j)) for i, j in sorted(bonds_ref)],
        "adsorbate_frame_bonds": [(int(i), int(j)) for i, j in sorted(bonds_now)],
        "removed_bonds": [(int(i), int(j)) for i, j in sorted(bonds_ref - bonds_now)],
        "added_bonds": [(int(i), int(j)) for i, j in sorted(bonds_now - bonds_ref)],
        "nearest_surface_by_ads_atom": pairwise,
        "adsorbate_pair_distances": bond_lengths,
    }


def run(args: argparse.Namespace) -> Path:
    suite = str(args.suite).strip().lower()
    registry = _crosslib_case_registry() if suite == "crosslib" else _miller_case_registry()
    case_name = str(args.case).strip()
    if case_name not in registry:
        raise KeyError(f"Unknown case: {case_name}")
    case = registry[case_name]
    slab = case["slab"]
    adsorbate = case["adsorbate"]
    slab_n = len(slab)
    case_root = _case_root(Path(args.benchmark_root), suite, case_name)
    relaxed_stream = case_root / "basin_work" / "relax" / "relaxed_stream.extxyz"
    frames = list(read(relaxed_stream.as_posix(), index=":"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    kept_frames = []
    rejected_frames = []
    rejected_by_reason: dict[str, list[Atoms]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    focus_ids = {int(x) for x in str(args.focus).split(",") if str(x).strip()} if str(args.focus).strip() else set()
    focus_payloads = []
    for idx, frame in enumerate(frames):
        reason, metrics = classify_anomaly(
            relaxed=frame,
            slab_ref=slab,
            adsorbate_ref=adsorbate,
            slab_n=int(slab_n),
            normal_axis=2,
            binding_tau=1.15,
            desorption_min_bonds=1,
            surface_reconstruction_max_disp=0.50,
            dissociation_allow_bond_change=False,
            burial_margin=0.30,
        )
        pairs = build_binding_pairs(frame, slab_n=int(slab_n), binding_tau=1.15)
        row = {
            "candidate_id": int(idx),
            "site_label": str(frame.info.get("site_label", "")),
            "basis_id": frame.info.get("basis_id", None),
            "reason": (None if reason is None else str(reason)),
            "metrics": dict(metrics),
            "binding_pairs": [(int(i), int(j)) for i, j in pairs],
        }
        candidate_rows.append(row)
        tagged = frame.copy()
        tagged.info["candidate_id"] = int(idx)
        tagged.info["reason"] = "" if reason is None else str(reason)
        for key, value in metrics.items():
            tagged.info[f"metric_{key}"] = value
        if reason is None:
            kept_frames.append(tagged)
        else:
            rejected_frames.append(tagged)
            rejected_by_reason[str(reason)].append(tagged)
            reason_counts[str(reason)] += 1
        if int(idx) in focus_ids:
            focus_payloads.append(_focus_payload(frame, idx=int(idx), slab_n=int(slab_n), adsorbate_ref=adsorbate))

    if kept_frames:
        write((out_root / "kept_frames.extxyz").as_posix(), kept_frames)
    if rejected_frames:
        write((out_root / "rejected_frames.extxyz").as_posix(), rejected_frames)
    for reason, frames_by_reason in sorted(rejected_by_reason.items()):
        safe = str(reason).replace("/", "_").replace(" ", "_")
        write((out_root / f"rejected_{safe}.extxyz").as_posix(), frames_by_reason)
    if focus_ids:
        focus_frames = [frames[int(i)].copy() for i in sorted(focus_ids) if 0 <= int(i) < len(frames)]
        if focus_frames:
            write((out_root / "focus_candidates.extxyz").as_posix(), focus_frames)
    payload = {
        "suite": suite,
        "case": case_name,
        "case_root": case_root.as_posix(),
        "relaxed_stream": relaxed_stream.as_posix(),
        "n_frames": int(len(frames)),
        "n_kept": int(len(kept_frames)),
        "n_rejected": int(len(rejected_frames)),
        "rejected_reason_counts": dict(reason_counts),
        "candidates": candidate_rows,
        "focus": focus_payloads,
    }
    _write_json(out_root / "anomaly_summary.json", payload)
    md_lines = [
        "# Case Anomaly Diagnosis",
        "",
        f"- Suite: {suite}",
        f"- Case: {case_name}",
        f"- Frames: {len(frames)}",
        f"- Kept: {len(kept_frames)}",
        f"- Rejected: {len(rejected_frames)}",
        "",
        "## Rejected Reasons",
        "",
    ]
    for reason, count in sorted(reason_counts.items()):
        md_lines.append(f"- {reason}: {int(count)}")
    if focus_payloads:
        md_lines.extend(["", "## Focus Candidates", ""])
        for fp in focus_payloads:
            md_lines.append(f"### Candidate {fp['candidate_id']}")
            md_lines.append("")
            md_lines.append(f"- Site label: {fp['site_label']}")
            md_lines.append(f"- Basis id: {fp['basis_id']}")
            md_lines.append(f"- Binding pairs: {fp['binding_pairs']}")
            md_lines.append(f"- Removed bonds: {fp['removed_bonds']}")
            md_lines.append(f"- Added bonds: {fp['added_bonds']}")
            md_lines.append("")
    (out_root / "anomaly_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print((out_root / "anomaly_summary.json").as_posix())
    print((out_root / "anomaly_summary.md").as_posix())
    return out_root / "anomaly_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True, choices=["crosslib", "miller"])
    parser.add_argument("--benchmark-root", type=str, required=True)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--focus", type=str, default="")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
