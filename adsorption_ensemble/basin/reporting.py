from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from ase import Atoms

from adsorption_ensemble.basin.pipeline import BasinBuilder
from adsorption_ensemble.basin.types import BasinConfig, BasinResult
from adsorption_ensemble.node.types import ReactionNode


def describe_basin_binding(*, basin_atoms: Atoms, slab_n: int, binding_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    ads = basin_atoms[int(slab_n) :].copy()
    slab = basin_atoms[: int(slab_n)].copy()
    ads_idx = sorted({int(i) for i, _ in binding_pairs})
    slab_idx = sorted({int(j) for _, j in binding_pairs})
    return {
        "binding_adsorbate_indices": [int(i) for i in ads_idx],
        "binding_adsorbate_symbols": [str(ads[int(i)].symbol) for i in ads_idx if 0 <= int(i) < len(ads)],
        "binding_surface_atom_ids": [int(j) for j in slab_idx],
        "binding_surface_symbols": [str(slab[int(j)].symbol) for j in slab_idx if 0 <= int(j) < len(slab)],
    }


def summarize_basin_member_provenance(member_frames: list[Atoms]) -> dict[str, Any]:
    site_labels = []
    basis_ids = []
    primitive_indices = []
    conformer_ids = []
    placement_modes = []
    for frame in member_frames:
        site_label = str(frame.info.get("site_label", "")).strip()
        if site_label:
            site_labels.append(site_label)
        basis_id = frame.info.get("basis_id", None)
        if basis_id is not None:
            try:
                basis_ids.append(int(basis_id))
            except Exception:
                pass
        primitive_index = frame.info.get("primitive_index", None)
        if primitive_index is not None:
            try:
                primitive_indices.append(int(primitive_index))
            except Exception:
                pass
        conformer_id = frame.info.get("conformer_id", None)
        if conformer_id is not None:
            try:
                conformer_ids.append(int(conformer_id))
            except Exception:
                pass
        placement_mode = str(frame.info.get("placement_mode", "")).strip()
        if placement_mode:
            placement_modes.append(placement_mode)
    return {
        "member_site_labels": sorted(set(site_labels)),
        "member_basis_ids": sorted(set(basis_ids)),
        "member_primitive_indices": sorted(set(primitive_indices)),
        "member_conformer_ids": sorted(set(conformer_ids)),
        "member_placement_modes": sorted(set(placement_modes)),
    }


def build_basin_dictionary(
    basin_result: BasinResult,
    *,
    pose_frames: list[Atoms] | None = None,
    nodes: list[ReactionNode] | None = None,
    slab_n: int | None = None,
) -> dict[str, Any]:
    basin_entries = []
    node_by_basin = {int(n.basin_id): n for n in nodes or []}
    for basin in basin_result.basins:
        member_frames = []
        if pose_frames is not None:
            for idx in basin.member_candidate_ids:
                if 0 <= int(idx) < len(pose_frames):
                    member_frames.append(pose_frames[int(idx)])
        energy_values = [float(basin.energy_ev)]
        energy_span = 0.0
        if member_frames:
            member_energies = []
            for frame in member_frames:
                e = frame.info.get("energy_ev", None)
                if e is not None:
                    try:
                        member_energies.append(float(e))
                    except Exception:
                        pass
            if member_energies:
                energy_values.extend(member_energies)
                energy_span = float(np.max(member_energies) - np.min(member_energies))
        rmsd_stats = _member_rmsd_stats(member_frames=member_frames, slab_n=slab_n)
        binding_meta = describe_basin_binding(
            basin_atoms=basin.atoms,
            slab_n=(0 if slab_n is None else int(slab_n)),
            binding_pairs=list(basin.binding_pairs),
        )
        provenance_meta = summarize_basin_member_provenance(member_frames)
        node = node_by_basin.get(int(basin.basin_id))
        basin_entries.append(
            {
                "basin_id": int(basin.basin_id),
                "signature": str(basin.signature),
                "energy_ev": float(basin.energy_ev),
                "denticity": int(basin.denticity),
                "member_candidate_ids": [int(x) for x in basin.member_candidate_ids],
                "member_count": int(len(basin.member_candidate_ids)),
                "binding_pairs": [(int(i), int(j)) for i, j in basin.binding_pairs],
                "energy_min_ev": float(np.nanmin(np.asarray(energy_values, dtype=float))),
                "energy_max_ev": float(np.nanmax(np.asarray(energy_values, dtype=float))),
                "energy_span_ev": float(energy_span),
                "member_adsorbate_rmsd_min": rmsd_stats["min"],
                "member_adsorbate_rmsd_max": rmsd_stats["max"],
                "member_adsorbate_rmsd_mean": rmsd_stats["mean"],
                "false_merge_suspect": bool(rmsd_stats["max"] is not None and rmsd_stats["max"] > 0.75),
                "node_id": (None if node is None else str(node.node_id)),
                **binding_meta,
                **provenance_meta,
            }
        )
    signature_groups: dict[str, int] = {}
    for entry in basin_entries:
        signature_groups[str(entry["signature"])] = signature_groups.get(str(entry["signature"]), 0) + 1
    false_split_suspects = [sig for sig, count in signature_groups.items() if int(count) > 1]
    return {
        "summary": dict(basin_result.summary),
        "relax_backend": str(basin_result.relax_backend),
        "n_basins": int(len(basin_entries)),
        "n_rejected": int(len(basin_result.rejected)),
        "false_split_suspect_signatures": sorted(false_split_suspects),
        "basins": basin_entries,
        "rejected": [
            {
                "candidate_id": int(r.candidate_id),
                "reason": str(r.reason),
                "metrics": dict(r.metrics),
            }
            for r in basin_result.rejected
        ],
    }


def run_basin_ablation(
    *,
    frames: list[Atoms],
    slab_ref: Atoms,
    adsorbate_ref: Atoms,
    slab_n: int,
    normal_axis: int,
    base_config: BasinConfig,
    relax_backend: object | None = None,
    metrics: tuple[str, ...] = ("signature_only", "rmsd"),
) -> dict[str, Any]:
    out: dict[str, Any] = {"metrics": {}}
    for metric in metrics:
        cfg = replace(base_config, dedup_metric=str(metric))
        try:
            result = BasinBuilder(config=cfg, relax_backend=relax_backend).build(
                frames=frames,
                slab_ref=slab_ref,
                adsorbate_ref=adsorbate_ref,
                slab_n=int(slab_n),
                normal_axis=int(normal_axis),
            )
            out["metrics"][str(metric)] = {
                "status": "ok",
                "summary": dict(result.summary),
                "n_basins": int(len(result.basins)),
                "n_rejected": int(len(result.rejected)),
                "basin_sizes": [int(len(b.member_candidate_ids)) for b in result.basins],
                "basin_signatures": [str(b.signature) for b in result.basins],
            }
        except Exception as exc:
            out["metrics"][str(metric)] = {
                "status": "error",
                "error_type": str(type(exc).__name__),
                "error_message": str(exc),
                "n_basins": 0,
                "n_rejected": 0,
                "basin_sizes": [],
                "basin_signatures": [],
            }
    counts = {k: int(v["n_basins"]) for k, v in out["metrics"].items()}
    if counts:
        min_metric = min(counts, key=counts.get)
        max_metric = max(counts, key=counts.get)
        out["comparison"] = {
            "min_basin_metric": str(min_metric),
            "max_basin_metric": str(max_metric),
            "basin_count_delta": int(counts[max_metric] - counts[min_metric]),
        }
    return out


def _member_rmsd_stats(member_frames: list[Atoms], slab_n: int | None) -> dict[str, float | None]:
    if slab_n is None or len(member_frames) <= 1:
        return {"min": None, "max": None, "mean": None}
    ref = np.asarray(member_frames[0].get_positions(), dtype=float)[int(slab_n) :]
    vals: list[float] = []
    for frame in member_frames[1:]:
        pos = np.asarray(frame.get_positions(), dtype=float)[int(slab_n) :]
        if pos.shape != ref.shape:
            continue
        vals.append(float(np.sqrt(np.mean(np.sum((pos - ref) ** 2, axis=1)))))
    if not vals:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}
