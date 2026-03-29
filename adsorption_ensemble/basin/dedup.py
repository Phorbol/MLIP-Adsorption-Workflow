from __future__ import annotations

import hashlib

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def build_adsorbate_bonds(adsorbate: Atoms, bond_tau: float = 1.20) -> set[tuple[int, int]]:
    pos = np.asarray(adsorbate.get_positions(), dtype=float)
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
    pairs = sorted(set(pairs))
    return pairs


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
    v, s, w = np.linalg.svd(c)
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
    diff = a - b
    return float(np.linalg.norm(diff, axis=1).sum())


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
    items: list[dict] = []
    for i, a in enumerate(frames):
        pairs = build_binding_pairs(a, slab_n=slab_n, binding_tau=binding_tau)
        sig = binding_signature(pairs)
        items.append(
            {
                "candidate_id": int(i),
                "atoms": a,
                "energy": float(energies[i]) if i < len(energies) else float("nan"),
                "binding_pairs": pairs,
                "signature": sig,
                "node_desc": (node_descriptors[i] if node_descriptors is not None and i < len(node_descriptors) else None),
            }
        )
    by_sig: dict[str, list[dict]] = {}
    for it in items:
        by_sig.setdefault(str(it["signature"]), []).append(it)
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        reps: list[dict] = []
        for it in group:
            assigned = False
            for rep in reps:
                da = it.get("node_desc")
                db = rep.get("node_desc")
                if da is None or db is None:
                    continue
                dist = sum_atomwise_l2(da, db)
                if np.isfinite(dist) and float(dist) <= float(node_l2_threshold):
                    rep["member_candidate_ids"].append(int(it["candidate_id"]))
                    assigned = True
                    break
            if not assigned:
                reps.append(
                    {
                        "basin_id": int(basin_id),
                        "atoms": it["atoms"],
                        "energy": float(it["energy"]),
                        "member_candidate_ids": [int(it["candidate_id"])],
                        "binding_pairs": it["binding_pairs"],
                        "signature": sig,
                        "node_desc": it.get("node_desc"),
                    }
                )
                basin_id += 1
        basins.extend(reps)
    for b in basins:
        b.pop("node_desc", None)
    return basins, meta


def cluster_by_signature_and_rmsd(
    frames: list[Atoms],
    energies: np.ndarray,
    slab_n: int,
    binding_tau: float,
    rmsd_threshold: float,
) -> list[dict]:
    items: list[dict] = []
    for i, a in enumerate(frames):
        pairs = build_binding_pairs(a, slab_n=slab_n, binding_tau=binding_tau)
        sig = binding_signature(pairs)
        items.append(
            {
                "candidate_id": int(i),
                "atoms": a,
                "energy": float(energies[i]) if i < len(energies) else float("nan"),
                "binding_pairs": pairs,
                "signature": sig,
            }
        )
    by_sig: dict[str, list[dict]] = {}
    for it in items:
        by_sig.setdefault(str(it["signature"]), []).append(it)
    basins: list[dict] = []
    basin_id = 0
    for sig, group in by_sig.items():
        group = sorted(group, key=lambda x: (np.nan_to_num(x["energy"], nan=np.inf), x["candidate_id"]))
        reps: list[dict] = []
        for it in group:
            pos = np.asarray(it["atoms"].get_positions(), dtype=float)[slab_n:]
            assigned = False
            for rep in reps:
                rep_pos = np.asarray(rep["atoms"].get_positions(), dtype=float)[slab_n:]
                rmsd = kabsch_rmsd(pos, rep_pos)
                if np.isfinite(rmsd) and float(rmsd) <= float(rmsd_threshold):
                    rep["member_candidate_ids"].append(int(it["candidate_id"]))
                    assigned = True
                    break
            if not assigned:
                reps.append(
                    {
                        "basin_id": int(basin_id),
                        "atoms": it["atoms"],
                        "energy": float(it["energy"]),
                        "member_candidate_ids": [int(it["candidate_id"])],
                        "binding_pairs": it["binding_pairs"],
                        "signature": sig,
                    }
                )
                basin_id += 1
        basins.extend(reps)
    return basins
