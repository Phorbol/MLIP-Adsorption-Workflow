from __future__ import annotations

import hashlib

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from adsorption_ensemble.basin.types import Basin
from adsorption_ensemble.node.types import NodeConfig, ReactionNode


def build_internal_bonds(adsorbate: Atoms, bond_tau: float) -> list[tuple[int, int]]:
    z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
    n = len(adsorbate)
    if n <= 1:
        return []
    d = adsorbate.get_all_distances(mic=False)
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        ri = float(covalent_radii[int(z[i])])
        for j in range(i + 1, n):
            rj = float(covalent_radii[int(z[j])])
            if float(d[i, j]) <= float(bond_tau) * (ri + rj):
                bonds.append((int(i), int(j)))
    return bonds


def canonicalize_adsorbate_order(atomic_numbers: list[int], bonds: list[tuple[int, int]], n_iter: int | None = None) -> list[int]:
    n = len(atomic_numbers)
    if n <= 1:
        return list(range(n))
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        if 0 <= i < n and 0 <= j < n and i != j:
            adj[i].append(j)
            adj[j].append(i)
    deg = [len(adj[i]) for i in range(n)]
    labels = [f"Z{int(atomic_numbers[i])}|D{int(deg[i])}" for i in range(n)]
    iters = int(n_iter) if n_iter is not None else min(6, max(2, n))
    for _ in range(iters):
        new_labels: list[str] = []
        for i in range(n):
            neigh = sorted(labels[j] for j in adj[i])
            s = labels[i] + "|" + "|".join(neigh)
            new_labels.append(hashlib.sha1(s.encode("utf-8")).hexdigest())
        labels = new_labels
    order = sorted(range(n), key=lambda i: (labels[i], int(atomic_numbers[i]), int(deg[i]), int(i)))
    return [int(i) for i in order]


def _remap_pairs(pairs: list[tuple[int, int]], inv: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in pairs:
        if 0 <= int(a) < len(inv):
            out.append((int(inv[int(a)]), int(b)))
    return sorted(set(out))


def _remap_bonds(bonds: list[tuple[int, int]], inv: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i, j in bonds:
        if 0 <= int(i) < len(inv) and 0 <= int(j) < len(inv):
            a = int(inv[int(i)])
            b = int(inv[int(j)])
            if a != b:
                out.append((min(a, b), max(a, b)))
    return sorted(set(out))


def make_node_id(
    atomic_numbers: list[int],
    internal_bonds: list[tuple[int, int]],
    binding_pairs: list[tuple[int, int]],
    hash_len: int = 20,
) -> str:
    parts = [
        "Z:" + ",".join(str(int(z)) for z in atomic_numbers),
        "B:" + ",".join(f"{i}-{j}" for i, j in internal_bonds),
        "S:" + ",".join(f"{i}:{j}" for i, j in binding_pairs),
    ]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[: max(8, int(hash_len))]


def make_node_id_from_surface_key(
    atomic_numbers: list[int],
    internal_bonds: list[tuple[int, int]],
    surface_key: str,
    hash_len: int = 20,
) -> str:
    parts = [
        "Z:" + ",".join(str(int(z)) for z in atomic_numbers),
        "B:" + ",".join(f"{i}-{j}" for i, j in internal_bonds),
        "K:" + str(surface_key),
    ]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[: max(8, int(hash_len))]


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


def _format_distance_list(values: list[float]) -> str:
    return ",".join(f"{float(v):.3f}" for v in values)


def _surface_pairwise_distances(surface_reference: Atoms, surface_ids: list[int]) -> list[float]:
    if len(surface_ids) <= 1:
        return []
    out: list[float] = []
    for i, a in enumerate(surface_ids):
        for b in surface_ids[i + 1 :]:
            out.append(float(surface_reference.get_distance(int(a), int(b), mic=True)))
    return sorted(out)


def _surface_env_key(
    *,
    binding_pairs: list[tuple[int, int]],
    surface_reference: Atoms,
    slab_n: int,
) -> str:
    env_pairs = []
    for ads_idx, surf_idx in sorted((int(i), int(j)) for i, j in binding_pairs):
        env = _surface_atom_environment_signature(
            frame=surface_reference,
            slab_n=int(slab_n),
            slab_atom_index=int(surf_idx),
        )
        env_pairs.append(f"{int(ads_idx)}:{env}")
    return hashlib.sha1("|".join(env_pairs).encode("utf-8")).hexdigest()[:16] if env_pairs else "free"


def _surface_geometry_key(
    *,
    binding_pairs: list[tuple[int, int]],
    surface_reference: Atoms,
    slab_n: int,
) -> str:
    pairs = [(int(i), int(j)) for i, j in binding_pairs]
    unique_surface_ids = sorted({int(j) for _, j in pairs})
    pairwise = _surface_pairwise_distances(surface_reference, unique_surface_ids)
    env_pairs = []
    for ads_idx, surf_idx in sorted(pairs):
        env = _surface_atom_environment_signature(
            frame=surface_reference,
            slab_n=int(slab_n),
            slab_atom_index=int(surf_idx),
        )
        peer_d = sorted(
            float(surface_reference.get_distance(int(surf_idx), int(other), mic=True))
            for other in unique_surface_ids
            if int(other) != int(surf_idx)
        )
        env_pairs.append(f"{int(ads_idx)}:{env}|peer:{_format_distance_list(peer_d)}")
    if not env_pairs:
        return "free"
    payload = [
        "S:" + "|".join(env_pairs),
        "G:" + _format_distance_list(pairwise),
    ]
    return hashlib.sha1("|".join(payload).encode("utf-8")).hexdigest()[:20]


def basin_to_node(
    basin: Basin,
    slab_n: int,
    cfg: NodeConfig | None = None,
    energy_min_ev: float | None = None,
    surface_reference: Atoms | None = None,
) -> ReactionNode:
    cfg_use = cfg or NodeConfig()
    ads = basin.atoms[int(slab_n) :].copy()
    z = [int(x) for x in np.asarray(ads.get_atomic_numbers(), dtype=int).tolist()]
    internal = build_internal_bonds(ads, bond_tau=float(cfg_use.bond_tau))
    order = canonicalize_adsorbate_order(z, internal)
    inv = [0] * len(order)
    for new_i, old_i in enumerate(order):
        inv[int(old_i)] = int(new_i)
    z_can = [z[i] for i in order]
    internal_can = _remap_bonds(internal, inv=inv)
    binding_can = _remap_pairs(list(basin.binding_pairs), inv=inv)
    node_id_legacy = make_node_id(
        atomic_numbers=z_can,
        internal_bonds=internal_can,
        binding_pairs=binding_can,
        hash_len=int(cfg_use.node_hash_len),
    )
    surface_ref_use = surface_reference.copy() if surface_reference is not None else basin.atoms[: int(slab_n)].copy()
    surface_env_key = _surface_env_key(
        binding_pairs=binding_can,
        surface_reference=surface_ref_use,
        slab_n=int(slab_n),
    )
    surface_geometry_key = _surface_geometry_key(
        binding_pairs=binding_can,
        surface_reference=surface_ref_use,
        slab_n=int(slab_n),
    )
    mode = str(cfg_use.node_identity_mode).strip().lower()
    if mode in {"surface_geometry", "surface-geometry", "geometry", "canonical_surface_geometry"}:
        node_id = make_node_id_from_surface_key(
            atomic_numbers=z_can,
            internal_bonds=internal_can,
            surface_key=str(surface_geometry_key),
            hash_len=int(cfg_use.node_hash_len),
        )
    else:
        node_id = str(node_id_legacy)
    dent = len({int(i) for i, _ in binding_can})
    rel_e = None
    if energy_min_ev is not None and np.isfinite(float(energy_min_ev)) and np.isfinite(float(basin.energy_ev)):
        rel_e = float(basin.energy_ev) - float(energy_min_ev)
    prov = {"basin_signature": str(basin.signature), "member_candidate_ids": list(basin.member_candidate_ids)}
    return ReactionNode(
        node_id=str(node_id),
        node_id_legacy=str(node_id_legacy),
        basin_id=int(basin.basin_id),
        canonical_order=[int(i) for i in order],
        atomic_numbers=[int(x) for x in z_can],
        internal_bonds=[(int(i), int(j)) for i, j in internal_can],
        binding_pairs=[(int(i), int(j)) for i, j in binding_can],
        surface_env_key=str(surface_env_key),
        surface_geometry_key=str(surface_geometry_key),
        denticity=int(dent),
        relative_energy_ev=rel_e,
        provenance=prov,
    )
