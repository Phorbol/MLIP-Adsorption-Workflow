from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from typing import Callable

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from scipy.optimize import linear_sum_assignment


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


def build_binding_pairs(
    frame: Atoms,
    slab_n: int,
    binding_tau: float,
    *,
    ignore_h_when_heavy_present: bool = True,
) -> list[tuple[int, int]]:
    n = len(frame)
    if slab_n <= 0 or slab_n >= n:
        return []
    z = np.asarray(frame.get_atomic_numbers(), dtype=int)
    d = frame.get_all_distances(mic=True)
    ads_z = np.asarray(z[slab_n:], dtype=int)
    has_heavy_adsorbate_atom = bool(np.any(ads_z != 1))
    pairs: list[tuple[int, int]] = []
    for ai in range(slab_n, n):
        if bool(ignore_h_when_heavy_present) and has_heavy_adsorbate_atom and int(z[ai]) == 1:
            continue
        ra = float(covalent_radii[int(z[ai])])
        for sj in range(0, slab_n):
            rs = float(covalent_radii[int(z[sj])])
            if float(d[ai, sj]) <= float(binding_tau) * (ra + rs):
                pairs.append((ai - slab_n, sj))
    return sorted(set(pairs))


def _surface_atom_environment_signature(frame: Atoms, slab_n: int, slab_atom_index: int, k: int = 6) -> str:
    if slab_n <= 1:
        return f"s{int(slab_atom_index)}"
    z = np.asarray(frame.get_atomic_numbers(), dtype=int)
    sym = int(z[int(slab_atom_index)])
    d = np.asarray(frame.get_distances(int(slab_atom_index), list(range(slab_n)), mic=True), dtype=float).reshape(-1)
    d = np.sort(d[d > 1e-8])
    take = d[: max(0, int(k))]
    rounded = ",".join(f"{float(v):.3f}" for v in take.tolist())
    return f"Z{sym}|{rounded}"


def binding_signature(
    binding_pairs: list[tuple[int, int]],
    *,
    frame: Atoms | None = None,
    slab_n: int | None = None,
    surface_reference: Atoms | None = None,
    mode: str = "absolute",
) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"none", "off", "disabled"}:
        return "all"
    if mode_norm in {"provenance", "basis", "site_label"}:
        if frame is None:
            raise ValueError("provenance binding_signature requires frame.")
        tokens = []
        site_label = str(frame.info.get("site_label", "")).strip()
        basis_id = frame.info.get("basis_id", None)
        for i, _ in sorted(binding_pairs):
            label = site_label if site_label else f"basis={basis_id}"
            tokens.append(f"{int(i)}:{label}")
        if not tokens:
            tokens.append(f"free:{site_label if site_label else f'basis={basis_id}'}")
        payload = ",".join(tokens).encode("utf-8")
    elif mode_norm in {"canonical", "equivalent", "site_equivalent"}:
        if frame is None or slab_n is None:
            raise ValueError("canonical binding_signature requires frame and slab_n.")
        tokens = []
        for i, j in sorted(binding_pairs):
            env = _surface_atom_environment_signature(frame=frame, slab_n=int(slab_n), slab_atom_index=int(j))
            tokens.append(f"{int(i)}:{env}")
        payload = ",".join(tokens).encode("utf-8")
    elif mode_norm in {"reference_canonical", "ref_canonical", "canonical_reference", "surface_reference_canonical"}:
        ref_frame = surface_reference
        if ref_frame is None or slab_n is None:
            raise ValueError("reference_canonical binding_signature requires surface_reference and slab_n.")
        tokens = []
        for i, j in sorted(binding_pairs):
            env = _surface_atom_environment_signature(frame=ref_frame, slab_n=int(slab_n), slab_atom_index=int(j))
            tokens.append(f"{int(i)}:{env}")
        payload = ",".join(tokens).encode("utf-8")
    else:
        payload = ",".join([f"{i}:{j}" for i, j in binding_pairs]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def binding_pattern_signature(binding_pairs: list[tuple[int, int]], *, frame: Atoms | None = None, slab_n: int | None = None) -> str:
    coord_by_adsorbate_atom: dict[int, int] = defaultdict(int)
    for i, _ in sorted(binding_pairs):
        coord_by_adsorbate_atom[int(i)] += 1
    tokens = []
    for ads_idx, coord in sorted(coord_by_adsorbate_atom.items()):
        symbol = ""
        if frame is not None and slab_n is not None:
            atom_idx = int(slab_n) + int(ads_idx)
            if 0 <= atom_idx < len(frame):
                symbol = str(frame[int(atom_idx)].symbol)
        if symbol:
            tokens.append(f"{int(ads_idx)}:{symbol}:{int(coord)}")
        else:
            tokens.append(f"{int(ads_idx)}:{int(coord)}")
    if not tokens:
        tokens.append("free")
    return hashlib.sha1(",".join(tokens).encode("utf-8")).hexdigest()[:16]


def local_binding_surface_descriptor(
    frame: Atoms,
    *,
    slab_n: int,
    binding_pairs: list[tuple[int, int]],
    k_nearest: int = 8,
    atom_mode: str = "binding_only",
    relative: bool = False,
) -> np.ndarray:
    n = len(frame)
    if int(slab_n) <= 0 or int(slab_n) >= n:
        return np.empty((0,), dtype=float)
    mode = str(atom_mode).strip().lower()
    binding_ads = sorted({int(i) for i, _ in binding_pairs})
    ads_z = np.asarray(frame.get_atomic_numbers(), dtype=int)[int(slab_n) :]
    if mode in {"binding_only", "binding_atoms", "bound"}:
        ads_indices = binding_ads
    elif mode in {"binding_heavy", "binding_non_h", "bound_heavy"}:
        ads_indices = [int(i) for i in binding_ads if 0 <= int(i) < len(ads_z) and int(ads_z[int(i)]) != 1]
        if not ads_indices:
            ads_indices = binding_ads
    elif mode in {"all_heavy", "heavy", "non_h"}:
        ads_indices = [int(i) for i, z in enumerate(ads_z.tolist()) if int(z) != 1]
        if not ads_indices:
            ads_indices = list(range(len(ads_z)))
    elif mode in {"all", "all_atoms", "adsorbate"}:
        ads_indices = list(range(len(ads_z)))
    else:
        raise ValueError(f"Unsupported atom_mode: {atom_mode}")
    if not ads_indices:
        return np.empty((0,), dtype=float)
    dmat = np.asarray(frame.get_all_distances(mic=True), dtype=float)
    k = max(1, int(k_nearest))
    feat = []
    for ai in ads_indices:
        if int(ai) < 0 or int(ai) >= len(ads_z):
            continue
        d = np.sort(np.asarray(dmat[int(slab_n) + int(ai), : int(slab_n)], dtype=float))
        d = d[: min(k, len(d))]
        if bool(relative) and d.size:
            d = d - float(d[0])
        feat.extend(float(v) for v in d.tolist())
    return np.asarray(feat, dtype=float)


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


def _wl_atom_classes(adsorbate: Atoms, *, bond_tau: float = 1.20, n_iter: int = 4) -> list[int]:
    bonds = build_adsorbate_bonds(adsorbate=adsorbate, bond_tau=float(bond_tau))
    n = len(adsorbate)
    if n <= 0:
        return []
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        nbrs[int(i)].append(int(j))
        nbrs[int(j)].append(int(i))
    z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
    labels = [f"Z{int(z[i])}|deg{len(nbrs[i])}" for i in range(n)]
    for _ in range(max(1, int(n_iter))):
        new_labels = []
        for i in range(n):
            neigh = sorted(labels[j] for j in nbrs[i])
            new_labels.append(f"{labels[i]}|{'/'.join(neigh)}")
        labels = new_labels
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return [int(unique[label]) for label in labels]


def _atom_class_groups(adsorbate: Atoms, *, bond_tau: float = 1.20) -> list[list[int]]:
    classes = _wl_atom_classes(adsorbate=adsorbate, bond_tau=float(bond_tau))
    groups: dict[int, list[int]] = defaultdict(list)
    for i, cls in enumerate(classes):
        groups[int(cls)].append(int(i))
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def _permute_positions(pos: np.ndarray, groups: list[list[int]], permuted_groups: list[tuple[int, ...]]) -> np.ndarray:
    out = np.asarray(pos, dtype=float).copy()
    for base, perm in zip(groups, permuted_groups):
        out[np.asarray(base, dtype=int)] = np.asarray(pos, dtype=float)[np.asarray(list(perm), dtype=int)]
    return out


def symmetry_aware_kabsch_rmsd(
    p: np.ndarray,
    q: np.ndarray,
    adsorbate: Atoms,
    *,
    bond_tau: float = 1.20,
    max_exact_permutations: int = 256,
    approx_iters: int = 3,
) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or len(adsorbate) != p.shape[0]:
        return float("nan")
    groups = _atom_class_groups(adsorbate=adsorbate, bond_tau=float(bond_tau))
    if all(len(g) <= 1 for g in groups):
        return float(kabsch_rmsd(p, q))

    exact_budget = 1
    group_perms: list[list[tuple[int, ...]]] = []
    for g in groups:
        if len(g) <= 1:
            perms = [tuple(g)]
        else:
            exact_budget *= int(math.factorial(len(g)))
            perms = list(itertools.permutations(g)) if exact_budget <= int(max_exact_permutations) else []
        group_perms.append(perms)

    if exact_budget <= int(max_exact_permutations) and all(group_perms):
        best = float("inf")
        for permuted_groups in itertools.product(*group_perms):
            d = float(kabsch_rmsd(_permute_positions(p, groups, list(permuted_groups)), q))
            if np.isfinite(d) and d < best:
                best = d
        return best

    q_centered = q - np.mean(q, axis=0, keepdims=True)
    perm_indices = np.arange(len(adsorbate), dtype=int)
    p_perm = np.asarray(p, dtype=float).copy()
    best = float(kabsch_rmsd(p_perm, q))
    for _ in range(max(1, int(approx_iters))):
        p_centered = p_perm - np.mean(p_perm, axis=0, keepdims=True)
        c = p_centered.T @ q_centered
        v, _, w = np.linalg.svd(c)
        d = np.sign(np.linalg.det(v @ w))
        r = v @ np.diag([1.0, 1.0, float(d)]) @ w
        p_rot = p_centered @ r
        next_perm = perm_indices.copy()
        for g in groups:
            if len(g) <= 1:
                continue
            g_arr = np.asarray(g, dtype=int)
            cost = np.linalg.norm(
                p_rot[g_arr][:, None, :] - q_centered[g_arr][None, :, :],
                axis=2,
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            next_perm[g_arr[row_ind]] = perm_indices[g_arr[col_ind]]
        if np.array_equal(next_perm, perm_indices):
            break
        perm_indices = next_perm
        p_perm = np.asarray(p, dtype=float)[perm_indices]
        best = min(best, float(kabsch_rmsd(p_perm, q)))
    return best


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


def _extract_adsorbate_node_descriptor(
    frame: Atoms,
    node_desc: np.ndarray | None,
    *,
    slab_n: int,
) -> tuple[Atoms | None, np.ndarray | None]:
    if node_desc is None:
        return None, None
    arr = np.asarray(node_desc, dtype=float)
    if arr.ndim != 2:
        return None, None
    n_total = len(frame)
    ads_n = int(n_total) - int(slab_n)
    if ads_n < 0:
        return None, None
    if arr.shape[0] == n_total:
        arr_ads = np.asarray(arr[int(slab_n) :], dtype=float)
    elif arr.shape[0] == ads_n:
        arr_ads = np.asarray(arr, dtype=float)
    else:
        return None, None
    return frame[int(slab_n) :].copy(), arr_ads


def _canonicalize_adsorbate_node_descriptor(
    frame: Atoms,
    node_desc: np.ndarray | None,
    *,
    slab_n: int,
    bond_tau: float = 1.20,
) -> tuple[np.ndarray | None, list[list[int]] | None, tuple[int, ...] | None]:
    adsorbate, ads_desc = _extract_adsorbate_node_descriptor(frame=frame, node_desc=node_desc, slab_n=int(slab_n))
    if adsorbate is None or ads_desc is None:
        return None, None, None
    from adsorption_ensemble.node.canonicalize import build_internal_bonds, canonicalize_adsorbate_order

    z = [int(v) for v in np.asarray(adsorbate.get_atomic_numbers(), dtype=int).tolist()]
    bonds = build_internal_bonds(adsorbate=adsorbate, bond_tau=float(bond_tau))
    order = canonicalize_adsorbate_order(z, bonds, n_iter=None)
    arr = np.asarray(ads_desc, dtype=float)
    if len(order) != arr.shape[0]:
        return None, None, None
    arr_can = np.asarray(arr[np.asarray(order, dtype=int)], dtype=float)
    ads_can = adsorbate[np.asarray(order, dtype=int)]
    groups = _atom_class_groups(ads_can, bond_tau=float(bond_tau))
    z_can = tuple(int(v) for v in np.asarray(ads_can.get_atomic_numbers(), dtype=int).tolist())
    return arr_can, groups, z_can


def _assigned_adsorbate_node_l2(
    a_desc: np.ndarray | None,
    a_groups: list[list[int]] | None,
    a_z: tuple[int, ...] | None,
    b_desc: np.ndarray | None,
    b_groups: list[list[int]] | None,
    b_z: tuple[int, ...] | None,
    *,
    l2_mode: str,
) -> float:
    if a_desc is None or b_desc is None or a_groups is None or b_groups is None or a_z is None or b_z is None:
        return float("nan")
    xa = np.asarray(a_desc, dtype=float)
    xb = np.asarray(b_desc, dtype=float)
    if xa.shape != xb.shape or xa.ndim != 2 or tuple(a_z) != tuple(b_z) or len(a_groups) != len(b_groups):
        return float("nan")
    matched_costs: list[float] = []
    for ga, gb in zip(a_groups, b_groups):
        if len(ga) != len(gb):
            return float("nan")
        ga_arr = np.asarray(ga, dtype=int)
        gb_arr = np.asarray(gb, dtype=int)
        block_a = np.asarray(xa[ga_arr], dtype=float)
        block_b = np.asarray(xb[gb_arr], dtype=float)
        if block_a.shape != block_b.shape:
            return float("nan")
        if block_a.shape[0] <= 1:
            matched_costs.extend(np.linalg.norm(block_a - block_b, axis=1).tolist())
            continue
        cost = np.linalg.norm(block_a[:, None, :] - block_b[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_costs.extend(cost[row_ind, col_ind].tolist())
    if not matched_costs:
        return 0.0
    total = float(np.sum(np.asarray(matched_costs, dtype=float)))
    mode_norm = str(l2_mode).strip().lower()
    if mode_norm in {"sum", "sum_atom", "sum_atomwise"}:
        return total
    return float(total / max(1, len(matched_costs)))


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


def _group_items_by_signature(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    *,
    signature_mode: str = "absolute",
    surface_reference: Atoms | None = None,
) -> dict[str, list[dict]]:
    items: list[dict] = []
    for i, atoms in enumerate(frames):
        pairs = build_binding_pairs(atoms, slab_n=slab_n, binding_tau=binding_tau)
        sig = binding_signature(
            pairs,
            frame=atoms,
            slab_n=slab_n,
            surface_reference=surface_reference,
            mode=str(signature_mode),
        )
        items.append(
            {
                "candidate_id": int(i),
                "atoms": atoms,
                "energy": float(energies[i]) if i < len(energies) else float("nan"),
                "binding_pairs": pairs,
                "signature": sig,
            }
        )
    by_sig: dict[str, list[dict]] = {}
    for it in items:
        by_sig.setdefault(str(it["signature"]), []).append(it)
    return by_sig


def _group_items_by_binding_pattern(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    *,
    surface_nearest_k: int,
    surface_atom_mode: str,
    surface_relative: bool,
) -> dict[str, list[dict]]:
    items: list[dict] = []
    for i, atoms in enumerate(frames):
        pairs = build_binding_pairs(atoms, slab_n=slab_n, binding_tau=binding_tau)
        pattern_sig = binding_pattern_signature(pairs, frame=atoms, slab_n=slab_n)
        surface_desc = local_binding_surface_descriptor(
            atoms,
            slab_n=int(slab_n),
            binding_pairs=pairs,
            k_nearest=int(surface_nearest_k),
            atom_mode=str(surface_atom_mode),
            relative=bool(surface_relative),
        )
        items.append(
            {
                "candidate_id": int(i),
                "atoms": atoms,
                "energy": float(energies[i]) if i < len(energies) else float("nan"),
                "binding_pairs": pairs,
                "binding_pattern_signature": str(pattern_sig),
                "surface_desc": surface_desc,
            }
        )
    by_sig: dict[str, list[dict]] = {}
    for it in items:
        by_sig.setdefault(str(it["binding_pattern_signature"]), []).append(it)
    return by_sig


def _descriptor_signature(pattern_sig: str, surface_desc: np.ndarray | None) -> str:
    if surface_desc is None:
        return hashlib.sha1(str(pattern_sig).encode("utf-8")).hexdigest()[:16]
    desc = np.asarray(surface_desc, dtype=float).reshape(-1)
    rounded = ",".join(f"{float(v):.3f}" for v in desc.tolist())
    return hashlib.sha1(f"{pattern_sig}|{rounded}".encode("utf-8")).hexdigest()[:16]


def _node_descriptor_signature(node_desc: np.ndarray | None, *, fallback_signature: str = "") -> str:
    if node_desc is None:
        payload = str(fallback_signature).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]
    desc = np.asarray(node_desc, dtype=float).reshape(-1)
    rounded = ",".join(f"{float(v):.4f}" for v in desc.tolist())
    return hashlib.sha1(rounded.encode("utf-8")).hexdigest()[:16]


def cluster_by_binding_pattern_and_surface_distance(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    surface_distance_threshold: float,
    surface_nearest_k: int = 8,
    surface_atom_mode: str = "binding_only",
    surface_relative: bool = False,
    surface_rmsd_gate: float | None = None,
    cluster_method: str = "greedy",
    fuzzy_sigma_scale: float = 1.5,
    fuzzy_membership_cutoff: float = 0.5,
) -> tuple[list[dict], dict]:
    by_pattern = _group_items_by_binding_pattern(
        frames=frames,
        energies=energies,
        slab_n=slab_n,
        binding_tau=binding_tau,
        surface_nearest_k=int(surface_nearest_k),
        surface_atom_mode=str(surface_atom_mode),
        surface_relative=bool(surface_relative),
    )
    basins: list[dict] = []
    basin_id = 0
    for pattern_sig, group in by_pattern.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))

        def distance_fn(a: dict, b: dict) -> float:
            da = np.asarray(a.get("surface_desc", np.empty((0,), dtype=float)), dtype=float).reshape(-1)
            db = np.asarray(b.get("surface_desc", np.empty((0,), dtype=float)), dtype=float).reshape(-1)
            if da.shape != db.shape:
                return float("nan")
            if surface_rmsd_gate is not None and np.isfinite(float(surface_rmsd_gate)):
                pa = np.asarray(a["atoms"].get_positions(), dtype=float)[int(slab_n) :]
                pb = np.asarray(b["atoms"].get_positions(), dtype=float)[int(slab_n) :]
                rr = float(symmetry_aware_kabsch_rmsd(pa, pb, a["atoms"][int(slab_n) :]))
                if not np.isfinite(rr) or rr > float(surface_rmsd_gate):
                    return float("inf")
            return float(np.linalg.norm(da - db))

        clusters = _cluster_group_by_distance(
            group=group,
            distance_fn=distance_fn,
            threshold=float(surface_distance_threshold),
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
                    "signature": _descriptor_signature(pattern_sig=str(pattern_sig), surface_desc=rep.get("surface_desc")),
                }
            )
            basin_id += 1
    meta = {
        "surface_distance_threshold": float(surface_distance_threshold),
        "surface_nearest_k": int(surface_nearest_k),
        "surface_atom_mode": str(surface_atom_mode),
        "surface_relative": bool(surface_relative),
        "surface_rmsd_gate": (None if surface_rmsd_gate is None else float(surface_rmsd_gate)),
        "cluster_method": str(cluster_method),
    }
    return basins, meta


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
    signature_mode: str = "absolute",
    use_signature_grouping: bool = True,
    surface_reference: Atoms | None = None,
    mace_enable_cueq: bool = False,
    energy_gate_ev: float | None = None,
) -> tuple[list[dict], dict]:
    if node_descriptors is None:
        from adsorption_ensemble.relax.backends import normalize_mace_descriptor_config
        from adsorption_ensemble.conformer_md.config import MACEInferenceConfig
        from adsorption_ensemble.conformer_md.mace_inference import MACEBatchInferencer

        model_path_use, device_use, dtype_use = normalize_mace_descriptor_config(
            model_path=mace_model_path,
            device=str(mace_device),
            dtype=str(mace_dtype),
            strict=False,
        )
        cfg = MACEInferenceConfig(
            model_path=str(model_path_use) if model_path_use else None,
            device=str(device_use),
            dtype=str(dtype_use),
            enable_cueq=bool(mace_enable_cueq),
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
    by_sig = _group_items_by_signature(
        frames=frames,
        energies=energies,
        slab_n=slab_n,
        binding_tau=binding_tau,
        signature_mode=str(signature_mode),
        surface_reference=surface_reference,
    )
    if not bool(use_signature_grouping):
        flat = []
        for group in by_sig.values():
            flat.extend(group)
        by_sig = {"all": flat}
    # Inject node descriptors after signature construction to keep logic centralized.
    flat_items = []
    for group in by_sig.values():
        flat_items.extend(group)
    for it in flat_items:
        cid = int(it["candidate_id"])
        it["node_desc"] = (node_descriptors[cid] if node_descriptors is not None and cid < len(node_descriptors) else None)
        ads_desc_can, ads_groups, ads_z = _canonicalize_adsorbate_node_descriptor(
            frame=it["atoms"],
            node_desc=it.get("node_desc"),
            slab_n=int(slab_n),
        )
        it["ads_node_desc"] = ads_desc_can
        it["ads_node_groups"] = ads_groups
        it["ads_atomic_numbers"] = ads_z

    l2_mode_norm = str(l2_mode).strip().lower()
    energy_gate = None
    if energy_gate_ev is not None and np.isfinite(float(energy_gate_ev)):
        energy_gate = float(energy_gate_ev)
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        clusters = _cluster_group_by_distance(
            group=group,
            distance_fn=lambda a, b: (
                float("inf")
                if (
                    energy_gate is not None
                    and (
                        not np.isfinite(float(a.get("energy", float("nan"))))
                        or not np.isfinite(float(b.get("energy", float("nan"))))
                        or abs(float(a["energy"]) - float(b["energy"])) > energy_gate
                    )
                )
                else (
                    float("nan")
                    if a.get("ads_node_desc") is None or b.get("ads_node_desc") is None
                    else float(
                        _assigned_adsorbate_node_l2(
                            a.get("ads_node_desc"),
                            a.get("ads_node_groups"),
                            a.get("ads_atomic_numbers"),
                            b.get("ads_node_desc"),
                            b.get("ads_node_groups"),
                            b.get("ads_atomic_numbers"),
                            l2_mode=str(l2_mode_norm),
                        )
                    )
                )
            ),
            threshold=float(node_l2_threshold),
            cluster_method=str(cluster_method),
            fuzzy_sigma_scale=float(fuzzy_sigma_scale),
            fuzzy_membership_cutoff=float(fuzzy_membership_cutoff),
        )
        for members in clusters:
            rep = sorted(members, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))[0]
            basin_signature = str(sig)
            if (not bool(use_signature_grouping)) or str(sig) == "all":
                basin_signature = _node_descriptor_signature(
                    rep.get("node_desc"),
                    fallback_signature="|".join(sorted(set(str(x.get("signature", "")) for x in members))),
                )
            basins.append(
                {
                    "basin_id": int(basin_id),
                    "atoms": rep["atoms"],
                    "energy": float(rep["energy"]),
                    "member_candidate_ids": [int(x["candidate_id"]) for x in members],
                    "binding_pairs": rep["binding_pairs"],
                    "signature": basin_signature,
                    "node_desc": rep.get("node_desc"),
                }
            )
            basin_id += 1
    for basin in basins:
        basin.pop("node_desc", None)
    meta["cluster_method"] = str(cluster_method)
    meta["l2_mode"] = str(l2_mode_norm)
    meta["node_l2_threshold"] = float(node_l2_threshold)
    meta["signature_mode"] = str(signature_mode)
    meta["use_signature_grouping"] = bool(use_signature_grouping)
    meta["descriptor_compare_mode"] = "adsorbate_canonical_assignment"
    meta["enable_cueq"] = bool(mace_enable_cueq)
    meta["energy_gate_ev"] = (None if energy_gate is None else float(energy_gate))
    return basins, meta


def merge_basin_representatives_by_mace_node_l2(
    basins: list[dict],
    *,
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
    cluster_method: str = "hierarchical",
    l2_mode: str = "mean_atom",
    fuzzy_sigma_scale: float = 1.5,
    fuzzy_membership_cutoff: float = 0.5,
    node_descriptors: list[np.ndarray | None] | None = None,
    signature_mode: str = "none",
    use_signature_grouping: bool = False,
    surface_reference: Atoms | None = None,
    mace_enable_cueq: bool = False,
    energy_gate_ev: float | None = None,
) -> tuple[list[dict], dict]:
    if not basins:
        return [], {
            "metric": "mace_node_l2",
            "n_input_basins": 0,
            "n_output_basins": 0,
            "node_l2_threshold": float(node_l2_threshold),
            "cluster_method": str(cluster_method),
            "signature_mode": str(signature_mode),
            "use_signature_grouping": bool(use_signature_grouping),
            "enable_cueq": bool(mace_enable_cueq),
            "energy_gate_ev": (None if energy_gate_ev is None else float(energy_gate_ev)),
        }
    rep_frames = [dict(b)["atoms"] for b in basins]
    rep_energies = np.asarray(
        [float(dict(b).get("energy", float("nan"))) for b in basins],
        dtype=float,
    )
    merged_reps, meta = cluster_by_signature_and_mace_node_l2(
        frames=rep_frames,
        energies=rep_energies,
        slab_n=int(slab_n),
        binding_tau=float(binding_tau),
        node_l2_threshold=float(node_l2_threshold),
        mace_model_path=mace_model_path,
        mace_device=str(mace_device),
        mace_dtype=str(mace_dtype),
        mace_enable_cueq=bool(mace_enable_cueq),
        mace_max_edges_per_batch=int(mace_max_edges_per_batch),
        mace_layers_to_keep=int(mace_layers_to_keep),
        mace_head_name=mace_head_name,
        mace_mlp_energy_key=mace_mlp_energy_key,
        cluster_method=str(cluster_method),
        l2_mode=str(l2_mode),
        fuzzy_sigma_scale=float(fuzzy_sigma_scale),
        fuzzy_membership_cutoff=float(fuzzy_membership_cutoff),
        node_descriptors=node_descriptors,
        signature_mode=str(signature_mode),
        use_signature_grouping=bool(use_signature_grouping),
        surface_reference=surface_reference,
        energy_gate_ev=energy_gate_ev,
    )
    merged_basins: list[dict] = []
    for merged in merged_reps:
        source_local_ids = sorted(set(int(x) for x in merged.get("member_candidate_ids", [])))
        source_basins = [dict(basins[i]) for i in source_local_ids if 0 <= int(i) < len(basins)]
        merged_candidate_ids = sorted(
            {
                int(cid)
                for basin in source_basins
                for cid in list(dict(basin).get("member_candidate_ids", []))
            }
        )
        source_signatures = sorted({str(dict(basin).get("signature", "")) for basin in source_basins})
        source_basin_ids = [
            int(dict(basin).get("basin_id", idx))
            for idx, basin in zip(source_local_ids, source_basins, strict=False)
        ]
        merged_basins.append(
            {
                "basin_id": int(merged["basin_id"]),
                "atoms": merged["atoms"],
                "energy": float(merged["energy"]),
                "member_candidate_ids": merged_candidate_ids,
                "binding_pairs": list(merged["binding_pairs"]),
                "signature": str(merged["signature"]),
                "source_basin_ids": source_basin_ids,
                "source_signatures": source_signatures,
            }
        )
    meta = dict(meta)
    meta["metric"] = "mace_node_l2"
    meta["n_input_basins"] = int(len(basins))
    meta["n_output_basins"] = int(len(merged_basins))
    meta["signature_mode"] = str(signature_mode)
    meta["use_signature_grouping"] = bool(use_signature_grouping)
    meta["enable_cueq"] = bool(mace_enable_cueq)
    meta["energy_gate_ev"] = (None if energy_gate_ev is None else float(energy_gate_ev))
    return merged_basins, meta


def cluster_by_signature_only(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    signature_mode: str = "absolute",
    surface_reference: Atoms | None = None,
) -> list[dict]:
    by_sig = _group_items_by_signature(
        frames=frames,
        energies=energies,
        slab_n=slab_n,
        binding_tau=binding_tau,
        signature_mode=str(signature_mode),
        surface_reference=surface_reference,
    )
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
    signature_mode: str = "absolute",
    use_signature_grouping: bool = True,
    surface_reference: Atoms | None = None,
) -> list[dict]:
    by_sig = _group_items_by_signature(
        frames=frames,
        energies=energies,
        slab_n=slab_n,
        binding_tau=binding_tau,
        signature_mode=str(signature_mode),
        surface_reference=surface_reference,
    )
    if not bool(use_signature_grouping):
        flat = []
        for group in by_sig.values():
            flat.extend(group)
        by_sig = {"all": flat}
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        clusters = _cluster_group_by_distance(
            group=group,
            distance_fn=lambda a, b: symmetry_aware_kabsch_rmsd(
                np.asarray(a["atoms"].get_positions(), dtype=float)[slab_n:],
                np.asarray(b["atoms"].get_positions(), dtype=float)[slab_n:],
                a["atoms"][slab_n:],
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
