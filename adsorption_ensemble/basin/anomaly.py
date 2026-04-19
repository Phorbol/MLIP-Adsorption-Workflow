from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from adsorption_ensemble.basin.dedup import build_adsorbate_bonds, build_binding_pairs


def classify_anomaly(
    relaxed: Atoms,
    slab_ref: Atoms,
    adsorbate_ref: Atoms,
    slab_n: int,
    normal_axis: int,
    binding_tau: float,
    desorption_min_bonds: int,
    desorption_contact_slack: float,
    surface_reconstruction_max_disp: float,
    dissociation_allow_bond_change: bool,
    burial_margin: float,
    surface_reconstruction_enabled: bool = True,
) -> tuple[str | None, dict]:
    metrics: dict = {}
    if slab_n <= 0 or slab_n >= len(relaxed):
        return "invalid_split", {"slab_n": int(slab_n), "n_atoms": int(len(relaxed))}
    binding_pairs = build_binding_pairs(relaxed, slab_n=slab_n, binding_tau=float(binding_tau))
    metrics["binding_pair_n"] = int(len(binding_pairs))
    if int(len(binding_pairs)) < int(desorption_min_bonds):
        slab = relaxed[:slab_n]
        ads = relaxed[slab_n:]
        surf_ids = list(range(int(slab_n)))
        min_ratio = float(np.inf)
        if len(ads) > 0 and len(surf_ids) > 0:
            ads_z = ads.get_atomic_numbers()
            surf_z = slab.get_atomic_numbers()
            surf_r = np.asarray([covalent_radii[int(z)] for z in surf_z], dtype=float)
            combined = slab + ads
            n_slab = len(slab)
            for i, z_ads in enumerate(ads_z):
                r_ads = float(covalent_radii[int(z_ads)])
                ci = n_slab + i
                d = np.asarray(combined.get_distances(ci, surf_ids, mic=True), dtype=float).reshape(-1)
                denom = np.maximum(1e-12, (r_ads + surf_r))
                if d.size > 0:
                    min_ratio = min(min_ratio, float(np.min(d / denom)))
        metrics["min_surface_contact_ratio"] = float(min_ratio)
        if not np.isfinite(min_ratio) or float(min_ratio) > float(binding_tau + desorption_contact_slack):
            return "desorption", metrics

    slab_pos0 = np.asarray(slab_ref.get_positions(), dtype=float)
    slab_pos1 = np.asarray(relaxed.get_positions(), dtype=float)[:slab_n]
    if slab_pos0.shape == slab_pos1.shape and slab_pos0.size > 0:
        disp = np.linalg.norm(slab_pos1 - slab_pos0, axis=1)
        metrics["surface_max_disp"] = float(np.max(disp))
        if bool(surface_reconstruction_enabled) and float(metrics["surface_max_disp"]) > float(surface_reconstruction_max_disp):
            return "surface_reconstruction", metrics

    ads0 = adsorbate_ref.copy()
    ads1 = relaxed[slab_n:].copy()
    b0 = build_adsorbate_bonds(ads0)
    b1 = build_adsorbate_bonds(ads1)
    metrics["ads_bond_n0"] = int(len(b0))
    metrics["ads_bond_n1"] = int(len(b1))
    if not bool(dissociation_allow_bond_change) and b0 != b1:
        metrics["ads_bond_changed"] = True
        metrics["bond_removed_n"] = int(len(b0 - b1))
        metrics["bond_added_n"] = int(len(b1 - b0))
        return "dissociation", metrics

    slab_coords = slab_pos1[:, int(normal_axis)]
    ads_coords = np.asarray(ads1.get_positions(), dtype=float)[:, int(normal_axis)]
    if slab_coords.size > 0 and ads_coords.size > 0:
        metrics["slab_min_axis"] = float(np.min(slab_coords))
        metrics["slab_max_axis"] = float(np.max(slab_coords))
        metrics["ads_min_axis"] = float(np.min(ads_coords))
        metrics["ads_max_axis"] = float(np.max(ads_coords))
        if float(metrics["ads_min_axis"]) < float(metrics["slab_min_axis"]) - float(burial_margin):
            return "burial", metrics

    return None, metrics
