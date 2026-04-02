from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ase import Atoms

from adsorption_ensemble.selection.strategies import DualThresholdSelector, EnergyWindowFilter, FarthestPointSamplingSelector, RMSDSelector


@dataclass
class StageSelectionConfig:
    enabled: bool = False
    strategy: str = "none"
    max_candidates: int = 0
    energy_window_ev: float | None = None
    rmsd_threshold: float = 0.10
    cluster_threshold: float = 0.10
    cluster_method: str = "hierarchical"
    descriptor: str = "adsorbate_surface_distance"
    random_seed: int = 0


def apply_stage_selection(
    *,
    frames: list[Atoms],
    config: StageSelectionConfig | None,
    slab_n: int = 0,
    energies: np.ndarray | None = None,
) -> tuple[list[int], dict[str, Any]]:
    n = int(len(frames))
    if n == 0:
        return [], {"enabled": bool(config.enabled) if config is not None else False, "strategy": "none", "n_input": 0, "n_selected": 0}
    cfg = config or StageSelectionConfig(enabled=False)
    if not bool(cfg.enabled) or str(cfg.strategy).strip().lower() in {"none", "off", "disabled"}:
        return list(range(n)), {
            "enabled": bool(cfg.enabled),
            "strategy": str(cfg.strategy),
            "n_input": int(n),
            "n_selected": int(n),
            "selected_ids": list(range(n)),
        }
    ids = list(range(n))
    e = None if energies is None else np.asarray(energies, dtype=float).reshape(-1)
    if e is not None and len(e) != n:
        raise ValueError("energies length must match frames length.")
    feat = _extract_features(frames=frames, slab_n=int(slab_n), descriptor=str(cfg.descriptor))
    strategy = str(cfg.strategy).strip().lower()
    if strategy in {"fps", "iterative_fps"}:
        cand = _apply_energy_window(ids=ids, energies=e, delta_e=cfg.energy_window_ev)
        k = int(cfg.max_candidates) if int(cfg.max_candidates) > 0 else len(cand)
        selected = FarthestPointSamplingSelector(random_seed=int(cfg.random_seed)).select(
            features=feat,
            k=min(k, len(cand)),
            candidate_ids=cand,
        )
        diagnostics = {
            "enabled": True,
            "strategy": strategy,
            "descriptor": str(cfg.descriptor),
            "n_input": int(n),
            "n_energy_window_keep": int(len(cand)),
            "n_selected": int(len(selected)),
            "selected_ids": [int(i) for i in selected],
        }
        return selected, diagnostics
    if strategy in {"energy_rmsd_window", "molclus", "molclus_like"}:
        if e is None:
            cand = list(ids)
            selected = RMSDSelector(threshold=float(cfg.rmsd_threshold)).select(features=feat, candidate_ids=cand)
        else:
            cand = _apply_energy_window(ids=ids, energies=e, delta_e=cfg.energy_window_ev)
            selected = DualThresholdSelector(
                energy_window=EnergyWindowFilter(delta_e=float(cfg.energy_window_ev if cfg.energy_window_ev is not None else 1.0e9)),
                rmsd_selector=RMSDSelector(threshold=float(cfg.rmsd_threshold)),
            ).select(energies=e, features=feat, candidate_ids=cand)
        if int(cfg.max_candidates) > 0:
            if e is None:
                selected = sorted(selected)[: int(cfg.max_candidates)]
            else:
                selected = sorted(selected, key=lambda i: float(e[i]))[: int(cfg.max_candidates)]
        diagnostics = {
            "enabled": True,
            "strategy": strategy,
            "descriptor": str(cfg.descriptor),
            "n_input": int(n),
            "n_energy_window_keep": int(len(cand)),
            "n_selected": int(len(selected)),
            "energy_available": bool(e is not None),
            "selected_ids": [int(i) for i in selected],
        }
        return selected, diagnostics
    if strategy in {"cluster", "hierarchical", "fuzzy"}:
        cand = _apply_energy_window(ids=ids, energies=e, delta_e=cfg.energy_window_ev)
        selected, cluster_sizes = _cluster_select_representatives(
            features=feat,
            candidate_ids=cand,
            energies=e,
            threshold=(float(cfg.cluster_threshold) if cfg.cluster_threshold > 0 else float(cfg.rmsd_threshold)),
            method=("hierarchical" if strategy == "cluster" else strategy),
            max_candidates=int(cfg.max_candidates),
        )
        diagnostics = {
            "enabled": True,
            "strategy": strategy,
            "descriptor": str(cfg.descriptor),
            "n_input": int(n),
            "n_energy_window_keep": int(len(cand)),
            "n_selected": int(len(selected)),
            "n_clusters": int(len(cluster_sizes)),
            "cluster_sizes": [int(x) for x in cluster_sizes],
            "selected_ids": [int(i) for i in selected],
        }
        return selected, diagnostics
    raise ValueError(f"Unsupported stage selection strategy: {cfg.strategy}")


def stage_selection_summary(config: StageSelectionConfig | None) -> dict[str, Any]:
    return asdict(config or StageSelectionConfig(enabled=False))


def _extract_features(frames: list[Atoms], slab_n: int, descriptor: str) -> np.ndarray:
    mode = str(descriptor).strip().lower()
    if mode in {"adsorbate_surface_distance", "surface_distance", "adsorption_surface_distance"}:
        if int(slab_n) <= 0:
            raise ValueError("adsorbate_surface_distance descriptor requires slab_n > 0.")
        return _adsorbate_surface_distance_features(frames=frames, slab_n=int(slab_n))
    if mode not in {"adsorbate_pair_distance", "pair_distance", "geometry_pair_distance"}:
        raise ValueError(f"Unsupported descriptor: {descriptor}")
    adsorbates = [a[int(slab_n) :].copy() if int(slab_n) > 0 else a.copy() for a in frames]
    if not adsorbates:
        return np.empty((0, 0), dtype=float)
    n_atoms = len(adsorbates[0])
    n_pairs = n_atoms * (n_atoms - 1) // 2
    out = np.zeros((len(adsorbates), n_pairs), dtype=float)
    for i, atoms in enumerate(adsorbates):
        if len(atoms) != n_atoms:
            raise ValueError("All frames must have identical adsorbate atom counts.")
        k = 0
        for a in range(n_atoms):
            for b in range(a + 1, n_atoms):
                out[i, k] = float(atoms.get_distance(a, b, mic=False))
                k += 1
    return out


def _adsorbate_surface_distance_features(*, frames: list[Atoms], slab_n: int) -> np.ndarray:
    if not frames:
        return np.empty((0, 0), dtype=float)
    first_ads_n = int(len(frames[0]) - int(slab_n))
    if first_ads_n <= 0:
        return np.empty((len(frames), 0), dtype=float)
    out = np.zeros((len(frames), int(slab_n) * first_ads_n), dtype=float)
    for i, atoms in enumerate(frames):
        if len(atoms) - int(slab_n) != first_ads_n:
            raise ValueError("All frames must have identical adsorbate atom counts.")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        slab_pos = positions[: int(slab_n)]
        ads_pos = positions[int(slab_n) :]
        k = 0
        for apos in ads_pos:
            dists = np.linalg.norm(slab_pos - apos[None, :], axis=1)
            sorted_dists = np.sort(np.asarray(dists, dtype=float))
            out[i, k : k + int(slab_n)] = sorted_dists
            k += int(slab_n)
    return out


def _apply_energy_window(ids: list[int], energies: np.ndarray | None, delta_e: float | None) -> list[int]:
    if energies is None or delta_e is None or not np.isfinite(float(delta_e)):
        return list(ids)
    return EnergyWindowFilter(delta_e=float(delta_e)).select(energies=np.asarray(energies, dtype=float), candidate_ids=list(ids))


def _cluster_select_representatives(
    *,
    features: np.ndarray,
    candidate_ids: list[int],
    energies: np.ndarray | None,
    threshold: float,
    method: str,
    max_candidates: int,
) -> tuple[list[int], list[int]]:
    if not candidate_ids:
        return [], []
    groups = _cluster_candidate_ids(
        features=features,
        candidate_ids=candidate_ids,
        threshold=float(threshold),
        method=str(method),
    )
    reps: list[int] = []
    cluster_sizes: list[int] = []
    for group in groups:
        cluster_sizes.append(int(len(group)))
        if energies is None:
            rep = min(group)
        else:
            rep = min(group, key=lambda i: (np.nan_to_num(float(energies[i]), nan=np.inf), int(i)))
        reps.append(int(rep))
    if energies is not None:
        reps = sorted(reps, key=lambda i: (np.nan_to_num(float(energies[i]), nan=np.inf), int(i)))
    else:
        reps = sorted(reps)
    if max_candidates > 0:
        reps = reps[:max_candidates]
    return reps, cluster_sizes


def _cluster_candidate_ids(
    *,
    features: np.ndarray,
    candidate_ids: list[int],
    threshold: float,
    method: str,
) -> list[list[int]]:
    ids = [int(i) for i in candidate_ids]
    n = len(ids)
    if n <= 1:
        return [ids] if ids else []
    x = np.asarray(features[np.asarray(ids, dtype=int)], dtype=float)
    dmat = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        dmat[i, i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(x[i] - x[j]))
            dmat[i, j] = d
            dmat[j, i] = d
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    mode = str(method).strip().lower()
    sigma = max(1e-8, float(threshold) * 1.5)
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(dmat[i, j])
            if not np.isfinite(dij):
                continue
            if mode in {"hierarchical", "agglomerative"}:
                keep = dij <= float(threshold)
            elif mode in {"fuzzy", "fuzzy_connected"}:
                keep = float(np.exp(-((dij / sigma) ** 2))) >= 0.5
            else:
                raise ValueError(f"Unsupported cluster method: {method}")
            if keep:
                union(i, j)
    out: dict[int, list[int]] = {}
    for i, cid in enumerate(ids):
        out.setdefault(find(i), []).append(int(cid))
    return [sorted(v) for _, v in sorted(out.items(), key=lambda kv: min(kv[1]))]
