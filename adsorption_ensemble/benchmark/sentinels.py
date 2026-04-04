from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from adsorption_ensemble.basin.dedup import build_binding_pairs


def summarize_adsorbate_binding_environment(
    frame: Atoms,
    *,
    slab_n: int,
    ads_atom_index: int = 0,
    nearest_k: int = 6,
    binding_tau: float = 1.15,
) -> dict[str, Any]:
    if int(slab_n) <= 0 or int(slab_n) >= len(frame):
        raise ValueError("slab_n must split slab and adsorbate atoms.")
    ads_idx = int(slab_n) + int(ads_atom_index)
    if ads_idx < int(slab_n) or ads_idx >= len(frame):
        raise IndexError("ads_atom_index is out of range for adsorbate atoms.")
    dmat = np.asarray(frame.get_all_distances(mic=True), dtype=float)
    slab_dist = np.asarray(dmat[ads_idx, : int(slab_n)], dtype=float)
    order = np.argsort(slab_dist)[: max(1, int(nearest_k))]
    nearest = [
        {
            "slab_atom_index": int(j),
            "symbol": str(frame[int(j)].symbol),
            "distance": float(slab_dist[int(j)]),
        }
        for j in order.tolist()
    ]
    binding_pairs = build_binding_pairs(frame, slab_n=int(slab_n), binding_tau=float(binding_tau))
    coordination = sum(1 for i, _ in binding_pairs if int(i) == int(ads_atom_index))
    return {
        "ads_atom_index": int(ads_atom_index),
        "ads_symbol": str(frame[ads_idx].symbol),
        "coordination": int(coordination),
        "binding_pairs": [(int(i), int(j)) for i, j in binding_pairs if int(i) == int(ads_atom_index)],
        "nearest_surface_neighbors": nearest,
    }


def audit_cu111_co_case(case_dir: str | Path) -> dict[str, Any]:
    root = Path(case_dir)
    ours_dir = root / "ours"
    selected_site_dict = json.loads((ours_dir / "selected_site_dictionary.json").read_text(encoding="utf-8"))
    basins_payload = json.loads((ours_dir / "basins.json").read_text(encoding="utf-8"))
    basin_frames = list(read((ours_dir / "basins.extxyz").as_posix(), index=":"))
    if not basin_frames:
        raise ValueError(f"No basins.extxyz frames found under {ours_dir}")
    slab_n = len(basin_frames[0]) - 2
    final_binding = summarize_adsorbate_binding_environment(
        basin_frames[0],
        slab_n=int(slab_n),
        ads_atom_index=0,
        nearest_k=6,
        binding_tau=1.15,
    )
    basis_sites = selected_site_dict.get("sites", [])
    site_labels = []
    for site in basis_sites:
        if isinstance(site, dict):
            site_labels.append(str(site.get("site_label", site.get("kind", ""))))
        else:
            site_labels.append(str(site))
    final_merge = dict(basins_payload.get("summary", {}).get("final_basin_merge", {}))
    return {
        "case_dir": root.as_posix(),
        "n_basis_sites": int(len(basis_sites)),
        "basis_site_labels": sorted(site_labels),
        "workflow_n_basins": int(basins_payload.get("summary", {}).get("n_basins", len(basin_frames))),
        "final_basin_merge": final_merge,
        "collapsed_all_basis_sites_into_one_basin": bool(len(basis_sites) > 1 and int(basins_payload.get("summary", {}).get("n_basins", 0)) == 1),
        "final_binding_environment": final_binding,
        "physics_flags": {
            "coordinated_to_three_or_more_surface_atoms": bool(int(final_binding["coordination"]) >= 3),
            "all_initial_sites_collapsed": bool(len(basis_sites) > 1 and int(basins_payload.get("summary", {}).get("n_basins", 0)) == 1),
        },
        "interpretation": (
            "suspicious_hollow_collapse"
            if bool(len(basis_sites) > 1 and int(basins_payload.get("summary", {}).get("n_basins", 0)) == 1 and int(final_binding["coordination"]) >= 3)
            else "no_hollow_collapse_flag"
        ),
    }
