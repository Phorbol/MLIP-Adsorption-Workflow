from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from ase import Atoms

from adsorption_ensemble.basin.pipeline import BasinBuilder
from adsorption_ensemble.basin.types import Basin, BasinConfig, BasinResult
from adsorption_ensemble.node import NodeConfig, basin_to_node
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
    surface_reference: Atoms | None = None,
    member_frames_relaxed: bool = False,
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
        rmsd_stats = (
            _member_rmsd_stats(member_frames=member_frames, slab_n=slab_n)
            if bool(member_frames_relaxed)
            else {"min": None, "max": None, "mean": None}
        )
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
                "false_merge_suspect": (None if rmsd_stats["max"] is None else bool(rmsd_stats["max"] > 0.75)),
                "node_id": (None if node is None else str(node.node_id)),
                "node_id_legacy": (None if node is None else str(node.node_id_legacy)),
                "surface_env_key": (None if node is None or node.surface_env_key is None else str(node.surface_env_key)),
                "surface_geometry_key": (
                    None if node is None or node.surface_geometry_key is None else str(node.surface_geometry_key)
                ),
                **binding_meta,
                **provenance_meta,
            }
        )
    signature_groups: dict[str, int] = {}
    for entry in basin_entries:
        signature_groups[str(entry["signature"])] = signature_groups.get(str(entry["signature"]), 0) + 1
    false_split_suspects = [sig for sig, count in signature_groups.items() if int(count) > 1]
    node_inflation_audit = None
    if nodes is not None and slab_n is not None and surface_reference is not None:
        node_inflation_audit = build_node_inflation_audit(
            basins=list(basin_result.basins),
            slab_n=int(slab_n),
            surface_reference=surface_reference,
            observed_nodes=list(nodes),
            node_cfg=NodeConfig(),
        )
    return {
        "summary": dict(basin_result.summary),
        "relax_backend": str(basin_result.relax_backend),
        "n_basins": int(len(basin_entries)),
        "n_rejected": int(len(basin_result.rejected)),
        "false_split_suspect_signatures": sorted(false_split_suspects),
        "node_inflation_audit": node_inflation_audit,
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


def run_named_basin_ablation(
    *,
    frames: list[Atoms],
    slab_ref: Atoms,
    adsorbate_ref: Atoms,
    slab_n: int,
    normal_axis: int,
    configs: dict[str, BasinConfig],
    relax_backend: object | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"configs": {}}
    counts: dict[str, int] = {}
    for name, cfg in configs.items():
        try:
            result = BasinBuilder(config=cfg, relax_backend=relax_backend).build(
                frames=frames,
                slab_ref=slab_ref,
                adsorbate_ref=adsorbate_ref,
                slab_n=int(slab_n),
                normal_axis=int(normal_axis),
            )
            out["configs"][str(name)] = {
                "status": "ok",
                "summary": dict(result.summary),
                "n_basins": int(len(result.basins)),
                "n_rejected": int(len(result.rejected)),
                "basin_sizes": [int(len(b.member_candidate_ids)) for b in result.basins],
                "basin_signatures": [str(b.signature) for b in result.basins],
                "basin_dictionary": build_basin_dictionary(result, pose_frames=frames, nodes=None, slab_n=int(slab_n)),
            }
            counts[str(name)] = int(len(result.basins))
        except Exception as exc:
            out["configs"][str(name)] = {
                "status": "error",
                "error_type": str(type(exc).__name__),
                "error_message": str(exc),
                "n_basins": 0,
                "n_rejected": 0,
                "basin_sizes": [],
                "basin_signatures": [],
                "basin_dictionary": {"summary": {}, "basins": [], "rejected": []},
            }
            counts[str(name)] = 0
    if counts:
        min_name = min(counts, key=counts.get)
        max_name = max(counts, key=counts.get)
        out["comparison"] = {
            "min_basin_config": str(min_name),
            "max_basin_config": str(max_name),
            "basin_count_delta": int(counts[max_name] - counts[min_name]),
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


def _node_field(node: ReactionNode | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _group_node_rows(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key_name]), []).append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda kv: (min(float(x["energy_ev"]) for x in kv[1]), kv[0])):
        energies = [float(x["energy_ev"]) for x in group]
        out.append(
            {
                "key": str(key),
                "count": int(len(group)),
                "basin_ids": [int(x["basin_id"]) for x in group],
                "node_ids": sorted({str(x["node_id"]) for x in group}),
                "node_id_legacies": sorted({str(x["node_id_legacy"]) for x in group}),
                "signatures": sorted({str(x["signature"]) for x in group}),
                "surface_env_keys": sorted({str(x["surface_env_key"]) for x in group}),
                "surface_geometry_keys": sorted({str(x["surface_geometry_key"]) for x in group}),
                "energy_min_ev": float(min(energies)),
                "energy_max_ev": float(max(energies)),
                "energy_span_ev": float(max(energies) - min(energies)),
            }
        )
    return out


def build_node_inflation_audit(
    *,
    basins: list[Basin],
    slab_n: int,
    surface_reference: Atoms,
    observed_nodes: list[ReactionNode | dict[str, Any]] | None = None,
    node_cfg: NodeConfig | None = None,
    compare_modes: tuple[str, ...] = ("legacy_absolute", "surface_geometry"),
) -> dict[str, Any]:
    cfg_use = node_cfg or NodeConfig()
    energy_min = (None if not basins else float(min(float(b.energy_ev) for b in basins)))

    observed_by_basin: dict[int, ReactionNode | dict[str, Any]] = {}
    for node in observed_nodes or []:
        basin_id = _node_field(node, "basin_id", None)
        if basin_id is None:
            continue
        observed_by_basin[int(basin_id)] = node

    rows_observed: list[dict[str, Any]] = []
    for basin in basins:
        obs = observed_by_basin.get(int(basin.basin_id))
        if obs is None:
            continue
        rows_observed.append(
            {
                "basin_id": int(basin.basin_id),
                "signature": str(basin.signature),
                "energy_ev": float(basin.energy_ev),
                "node_id": str(_node_field(obs, "node_id", "")),
                "node_id_legacy": str(_node_field(obs, "node_id_legacy", _node_field(obs, "node_id", ""))),
                "surface_env_key": str(_node_field(obs, "surface_env_key", "")),
                "surface_geometry_key": str(_node_field(obs, "surface_geometry_key", "")),
            }
        )

    rows_by_mode: dict[str, list[dict[str, Any]]] = {"observed": rows_observed}
    for mode in compare_modes:
        cfg_mode = NodeConfig(
            bond_tau=float(cfg_use.bond_tau),
            node_hash_len=int(cfg_use.node_hash_len),
            node_identity_mode=str(mode),
        )
        nodes_mode = [
            basin_to_node(
                basin,
                slab_n=int(slab_n),
                cfg=cfg_mode,
                energy_min_ev=energy_min,
                surface_reference=surface_reference,
            )
            for basin in basins
        ]
        rows_mode = []
        for basin, node in zip(basins, nodes_mode, strict=True):
            rows_mode.append(
                {
                    "basin_id": int(basin.basin_id),
                    "signature": str(basin.signature),
                    "energy_ev": float(basin.energy_ev),
                    "node_id": str(node.node_id),
                    "node_id_legacy": str(node.node_id_legacy),
                    "surface_env_key": ("" if node.surface_env_key is None else str(node.surface_env_key)),
                    "surface_geometry_key": ("" if node.surface_geometry_key is None else str(node.surface_geometry_key)),
                }
            )
        rows_by_mode[str(mode)] = rows_mode

    grouped_by_mode = {mode: _group_node_rows(rows, "node_id") for mode, rows in rows_by_mode.items()}
    observed_lookup = {int(r["basin_id"]): r for r in rows_observed}
    legacy_lookup = {int(r["basin_id"]): r for r in rows_by_mode.get("legacy_absolute", [])}
    split_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows_by_mode.get("surface_geometry", []):
        split_groups.setdefault(str(row["surface_geometry_key"]), []).append(row)

    redundant_splits = []
    for key, group in sorted(split_groups.items(), key=lambda kv: (min(float(x["energy_ev"]) for x in kv[1]), kv[0])):
        basin_ids = [int(x["basin_id"]) for x in group]
        observed_ids = sorted({str(observed_lookup[int(bid)]["node_id"]) for bid in basin_ids if int(bid) in observed_lookup})
        if len(observed_ids) <= 1:
            continue
        energies = [float(x["energy_ev"]) for x in group]
        redundant_splits.append(
            {
                "surface_geometry_key": str(key),
                "basin_ids": basin_ids,
                "observed_node_ids": observed_ids,
                "legacy_node_ids": sorted(
                    {str(legacy_lookup[int(bid)]["node_id"]) for bid in basin_ids if int(bid) in legacy_lookup}
                ),
                "signatures": sorted({str(x["signature"]) for x in group}),
                "energy_min_ev": float(min(energies)),
                "energy_max_ev": float(max(energies)),
                "energy_span_ev": float(max(energies) - min(energies)),
            }
        )

    observed_count = len(grouped_by_mode["observed"]) if rows_observed else 0
    legacy_count = len(grouped_by_mode.get("legacy_absolute", []))
    geom_count = len(grouped_by_mode.get("surface_geometry", []))
    redundant_split_basin_count = int(sum(len(group["basin_ids"]) for group in redundant_splits))
    return {
        "current_node_identity_mode": str(cfg_use.node_identity_mode),
        "counts": {
            "basins": int(len(basins)),
            "observed_node_ids": int(observed_count),
            "legacy_absolute": int(legacy_count),
            "surface_geometry": int(geom_count),
            "inflation_observed_vs_surface_geometry": int(observed_count - geom_count),
            "inflation_legacy_vs_surface_geometry": int(legacy_count - geom_count),
            "redundant_surface_geometry_split_groups": int(len(redundant_splits)),
            "redundant_surface_geometry_split_basins": int(redundant_split_basin_count),
        },
        "rows_by_mode": rows_by_mode,
        "groups": grouped_by_mode,
        "redundant_surface_geometry_splits": redundant_splits,
    }
