from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def build_adsorbate_bonds(adsorbate: Atoms, bond_tau: float = 1.20) -> set[tuple[int, int]]:
    z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
    n = len(adsorbate)
    if n <= 1:
        return set()
    d = adsorbate.get_all_distances(mic=False)
    bonds: set[tuple[int, int]] = set()
    for i in range(n):
        ri = float(covalent_radii[int(z[i])])
        for j in range(i + 1, n):
            rj = float(covalent_radii[int(z[j])])
            if float(d[i, j]) <= float(bond_tau) * (ri + rj):
                bonds.add((i, j))
    return bonds


def build_binding_pairs(frame: Atoms, slab_n: int, binding_tau: float) -> list[tuple[int, int]]:
    n = len(frame)
    if slab_n <= 0 or slab_n >= n:
        return []
    z = np.asarray(frame.get_atomic_numbers(), dtype=int)
    d = frame.get_all_distances(mic=True)
    pairs: list[tuple[int, int]] = []
    for ai in range(slab_n, n):
        ra = float(covalent_radii[int(z[ai])])
        for sj in range(0, slab_n):
            rs = float(covalent_radii[int(z[sj])])
            if float(d[ai, sj]) <= float(binding_tau) * (ra + rs):
                pairs.append((ai - slab_n, sj))
    return sorted(set(pairs))


def binding_signature(binding_pairs: list[tuple[int, int]]) -> str:
    payload = ",".join([f"{i}:{j}" for i, j in binding_pairs]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        return float("nan")
    pc = p - np.mean(p, axis=0, keepdims=True)
    qc = q - np.mean(q, axis=0, keepdims=True)
    c = pc.T @ qc
    v, _, w = np.linalg.svd(c)
    d = np.sign(np.linalg.det(v @ w))
    r = v @ np.diag([1.0, 1.0, float(d)]) @ w
    pr = pc @ r
    diff = pr - qc
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def sum_atomwise_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        return float("nan")
    return float(np.linalg.norm(a - b, axis=1).sum())


def mean_atomwise_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        return float("nan")
    return float(np.linalg.norm(a - b, axis=1).mean())


def _pairwise_distance_matrix(items: list[dict], distance_fn: Callable[[dict, dict], float]) -> np.ndarray:
    n = len(items)
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    out = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        out[i, i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance_fn(items[i], items[j]))
            out[i, j] = d
            out[j, i] = d
    return out


def _components_from_threshold_graph(dmat: np.ndarray, threshold: float) -> list[list[int]]:
    n = int(dmat.shape[0])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            dij = float(dmat[i, j])
            if np.isfinite(dij) and dij <= float(threshold):
                union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _components_from_fuzzy_affinity(
    dmat: np.ndarray,
    threshold: float,
    *,
    sigma_scale: float,
    membership_cutoff: float,
) -> list[list[int]]:
    n = int(dmat.shape[0])
    sigma = max(1e-8, float(threshold) * float(sigma_scale))
    alpha = float(np.clip(float(membership_cutoff), 1e-6, 1.0 - 1e-6))
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            dij = float(dmat[i, j])
            if not np.isfinite(dij):
                continue
            aff = float(np.exp(-((dij / sigma) ** 2)))
            if aff >= alpha:
                union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _cluster_group_by_distance(
    group: list[dict],
    *,
    distance_fn: Callable[[dict, dict], float],
    threshold: float,
    cluster_method: str,
    fuzzy_sigma_scale: float = 1.5,
    fuzzy_membership_cutoff: float = 0.5,
) -> list[list[dict]]:
    method = str(cluster_method).strip().lower()
    if method in {"greedy", "sequential"}:
        reps: list[dict] = []
        for it in group:
            assigned = False
            for rep in reps:
                d = float(distance_fn(it, rep))
                if np.isfinite(d) and d <= float(threshold):
                    rep["_members"].append(it)
                    assigned = True
                    break
            if not assigned:
                rep = dict(it)
                rep["_members"] = [it]
                reps.append(rep)
        return [list(rep["_members"]) for rep in reps]
    dmat = _pairwise_distance_matrix(items=group, distance_fn=distance_fn)
    if method in {"hierarchical", "agglomerative"}:
        idx_groups = _components_from_threshold_graph(dmat=dmat, threshold=float(threshold))
    elif method in {"fuzzy", "fuzzy_connected"}:
        idx_groups = _components_from_fuzzy_affinity(
            dmat=dmat,
            threshold=float(threshold),
            sigma_scale=float(fuzzy_sigma_scale),
            membership_cutoff=float(fuzzy_membership_cutoff),
        )
    else:
        raise ValueError(f"Unsupported cluster_method: {cluster_method}")
    return [[group[i] for i in sorted(g)] for g in idx_groups]


def _group_items_by_signature(frames: list[Atoms], energies: np.ndarray, slab_n: int, binding_tau: float) -> dict[str, list[dict]]:
    items: list[dict] = []
    for i, atoms in enumerate(frames):
        pairs = build_binding_pairs(atoms, slab_n=slab_n, binding_tau=binding_tau)
        items.append(
            {
                "candidate_id": int(i),
                "atoms": atoms,
                "energy": float(energies[i]) if i < len(energies) else float("nan"),
                "binding_pairs": pairs,
                "signature": binding_signature(pairs),
            }
        )
    by_sig: dict[str, list[dict]] = {}
    for it in items:
        by_sig.setdefault(str(it["signature"]), []).append(it)
    return by_sig


def cluster_by_signature_and_mace_node_l2(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    node_l2_threshold: float,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
    mace_max_edges_per_batch: int,
    mace_layers_to_keep: int,
    mace_head_name: str | None,
    mace_mlp_energy_key: str | None,
    cluster_method: str = "greedy",
    l2_mode: str = "mean_atom",
    fuzzy_sigma_scale: float = 1.5,
    fuzzy_membership_cutoff: float = 0.5,
    node_descriptors: list[np.ndarray | None] | None = None,
) -> tuple[list[dict], dict]:
    if node_descriptors is None:
        from adsorption_ensemble.conformer_md.config import MACEInferenceConfig
        from adsorption_ensemble.conformer_md.mace_inference import MACEBatchInferencer

        cfg = MACEInferenceConfig(
            model_path=str(mace_model_path) if mace_model_path else None,
            device=str(mace_device),
            dtype=str(mace_dtype),
            max_edges_per_batch=int(mace_max_edges_per_batch),
            num_workers=1,
            layers_to_keep=int(mace_layers_to_keep),
            mlp_energy_key=(str(mace_mlp_energy_key) if mace_mlp_energy_key else None),
            head_name=str(mace_head_name) if mace_head_name is not None and str(mace_head_name).strip() else "Default",
        )
        infer = MACEBatchInferencer(cfg)
        node_descriptors, _, meta = infer.infer_node_descriptors(frames)
    else:
        meta = {"provided_node_descriptors": True}
    by_sig = _group_items_by_signature(frames=frames, energies=energies, slab_n=slab_n, binding_tau=binding_tau)
    # Inject node descriptors after signature construction to keep logic centralized.
    flat_items = []
    for group in by_sig.values():
        flat_items.extend(group)
    for it in flat_items:
        cid = int(it["candidate_id"])
        it["node_desc"] = (node_descriptors[cid] if node_descriptors is not None and cid < len(node_descriptors) else None)

    l2_mode_norm = str(l2_mode).strip().lower()
    l2_fn = sum_atomwise_l2 if l2_mode_norm in {"sum", "sum_atom", "sum_atomwise"} else mean_atomwise_l2
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        clusters = _cluster_group_by_distance(
            group=group,
            distance_fn=lambda a, b: (
                float("nan")
                if a.get("node_desc") is None or b.get("node_desc") is None
                else float(l2_fn(a["node_desc"], b["node_desc"]))
            ),
            threshold=float(node_l2_threshold),
            cluster_method=str(cluster_method),
            fuzzy_sigma_scale=float(fuzzy_sigma_scale),
            fuzzy_membership_cutoff=float(fuzzy_membership_cutoff),
        )
        for members in clusters:
            rep = sorted(members, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))[0]
            basins.append(
                {
                    "basin_id": int(basin_id),
                    "atoms": rep["atoms"],
                    "energy": float(rep["energy"]),
                    "member_candidate_ids": [int(x["candidate_id"]) for x in members],
                    "binding_pairs": rep["binding_pairs"],
                    "signature": sig,
                    "node_desc": rep.get("node_desc"),
                }
            )
            basin_id += 1
    for basin in basins:
        basin.pop("node_desc", None)
    meta["cluster_method"] = str(cluster_method)
    meta["l2_mode"] = str(l2_mode_norm)
    meta["node_l2_threshold"] = float(node_l2_threshold)
    return basins, meta


def cluster_by_signature_only(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
) -> list[dict]:
    by_sig = _group_items_by_signature(frames=frames, energies=energies, slab_n=slab_n, binding_tau=binding_tau)
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        rep = group[0]
        basins.append(
            {
                "basin_id": int(basin_id),
                "atoms": rep["atoms"],
                "energy": float(rep["energy"]),
                "member_candidate_ids": [int(x["candidate_id"]) for x in group],
                "binding_pairs": rep["binding_pairs"],
                "signature": sig,
            }
        )
        basin_id += 1
    return basins


def cluster_by_signature_and_rmsd(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    rmsd_threshold: float,
    cluster_method: str = "greedy",
    fuzzy_sigma_scale: float = 1.5,
    fuzzy_membership_cutoff: float = 0.5,
) -> list[dict]:
    by_sig = _group_items_by_signature(frames=frames, energies=energies, slab_n=slab_n, binding_tau=binding_tau)
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        clusters = _cluster_group_by_distance(
            group=group,
            distance_fn=lambda a, b: kabsch_rmsd(
                np.asarray(a["atoms"].get_positions(), dtype=float)[slab_n:],
                np.asarray(b["atoms"].get_positions(), dtype=float)[slab_n:],
            ),
            threshold=float(rmsd_threshold),
            cluster_method=str(cluster_method),
            fuzzy_sigma_scale=float(fuzzy_sigma_scale),
            fuzzy_membership_cutoff=float(fuzzy_membership_cutoff),
        )
        for members in clusters:
            rep = sorted(members, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))[0]
            basins.append(
                {
                    "basin_id": int(basin_id),
                    "atoms": rep["atoms"],
                    "energy": float(rep["energy"]),
                    "member_candidate_ids": [int(x["candidate_id"]) for x in members],
                    "binding_pairs": rep["binding_pairs"],
                    "signature": sig,
                }
            )
            basin_id += 1
    return basins
