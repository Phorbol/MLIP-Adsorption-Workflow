from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from .primitives import SitePrimitive


@dataclass
class PrimitiveEmbeddingConfig:
    l2_distance_threshold: float = 0.30
    include_geom_aux: bool = True
    geom_k_nearest: int = 6


@dataclass
class PrimitiveEmbeddingResult:
    primitives: list[SitePrimitive]
    basis_primitives: list[SitePrimitive]
    bucket_sizes: dict[str, int]
    cluster_sizes: dict[int, int]
    raw_count: int
    basis_count: int
    compression_ratio: float


class PrimitiveEmbedder:
    def __init__(self, config: PrimitiveEmbeddingConfig | None = None):
        self.config = config or PrimitiveEmbeddingConfig()

    def fit_transform(
        self,
        slab: Atoms,
        primitives: list[SitePrimitive],
        atom_features: np.ndarray,
    ) -> PrimitiveEmbeddingResult:
        self._validate_inputs(slab, primitives, atom_features)
        for p in primitives:
            p.embedding = self._build_primitive_embedding(p, atom_features=atom_features, slab=slab, config=self.config)
        buckets = self._bucket_by_topology(primitives)
        assigned = self._cluster_within_buckets(primitives, buckets)
        basis_ids = sorted({int(x) for x in assigned if x is not None})
        basis_primitives = [primitives[i] for i in basis_ids]
        cluster_sizes: dict[int, int] = {i: 0 for i in basis_ids}
        for p in primitives:
            cluster_sizes[int(p.basis_id)] += 1
        raw_count = len(primitives)
        basis_count = len(basis_primitives)
        ratio = float(basis_count) / float(raw_count) if raw_count > 0 else 0.0
        return PrimitiveEmbeddingResult(
            primitives=primitives,
            basis_primitives=basis_primitives,
            bucket_sizes={k: len(v) for k, v in buckets.items()},
            cluster_sizes=cluster_sizes,
            raw_count=raw_count,
            basis_count=basis_count,
            compression_ratio=ratio,
        )

    @staticmethod
    def _validate_inputs(slab: Atoms, primitives: list[SitePrimitive], atom_features: np.ndarray) -> None:
        if atom_features.ndim != 2:
            raise ValueError("atom_features must be a 2D array with shape (n_atoms, feat_dim).")
        if atom_features.shape[0] != len(slab):
            raise ValueError("atom_features row count must equal slab atom count.")
        if not primitives:
            return
        feat_dim = atom_features.shape[1]
        if feat_dim <= 0:
            raise ValueError("atom_features must have non-zero feature dimension.")

    @staticmethod
    def _build_primitive_embedding(
        primitive: SitePrimitive,
        atom_features: np.ndarray,
        slab: Atoms,
        config: PrimitiveEmbeddingConfig,
    ) -> np.ndarray:
        idx = np.array(primitive.atom_ids, dtype=int)
        feats = atom_features[idx]
        mean_feat = np.mean(feats, axis=0)
        kind_map = {"1c": 0, "2c": 1, "3c": 2, "4c": 3}
        one_hot = np.zeros(4, dtype=float)
        one_hot[kind_map.get(str(primitive.kind), 0)] = 1.0
        if not bool(config.include_geom_aux):
            return np.asarray(np.concatenate([mean_feat, one_hot]), dtype=float)
        dfeat = PrimitiveEmbedder._center_distance_fingerprint(
            slab=slab,
            center=np.asarray(primitive.center, dtype=float),
            k=max(0, int(config.geom_k_nearest)),
        )
        return np.asarray(np.concatenate([mean_feat, one_hot, dfeat]), dtype=float)

    @staticmethod
    def _center_distance_fingerprint(slab: Atoms, center: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or len(slab) <= 0:
            return np.zeros(max(0, k), dtype=float)
        cell = np.asarray(slab.cell.array, dtype=float)
        pbc = slab.get_pbc()
        vec = np.asarray(slab.get_positions(), dtype=float) - center.reshape(1, 3)
        try:
            frac = np.linalg.solve(cell.T, vec.T).T
            for ax in range(3):
                if bool(pbc[ax]):
                    frac[:, ax] -= np.round(frac[:, ax])
            vec = frac @ cell
        except Exception:
            pass
        d = np.linalg.norm(vec, axis=1)
        d_sorted = np.sort(np.asarray(d, dtype=float))
        out = np.zeros(k, dtype=float)
        take = min(k, int(d_sorted.shape[0]))
        if take > 0:
            out[:take] = d_sorted[:take]
        nz = d_sorted[d_sorted > 1e-8]
        scale = float(nz[0]) if nz.size > 0 else 1.0
        if scale <= 1e-8:
            scale = 1.0
        return out / scale

    @staticmethod
    def _bucket_by_topology(primitives: list[SitePrimitive]) -> dict[str, list[int]]:
        buckets: dict[str, list[int]] = {}
        for i, p in enumerate(primitives):
            buckets.setdefault(p.topo_hash, []).append(i)
        for k in buckets:
            buckets[k] = sorted(buckets[k], key=lambda x: (primitives[x].kind, primitives[x].atom_ids))
        return buckets

    def _cluster_within_buckets(
        self,
        primitives: list[SitePrimitive],
        buckets: dict[str, list[int]],
    ) -> list[int | None]:
        assigned: list[int | None] = [None] * len(primitives)
        for idxs in buckets.values():
            centers: list[int] = []
            for i in idxs:
                if not centers:
                    assigned[i] = i
                    primitives[i].basis_id = i
                    centers.append(i)
                    continue
                matched = None
                for c in centers:
                    d = self._embedding_distance(primitives[i], primitives[c])
                    if d <= self.config.l2_distance_threshold:
                        matched = c
                        break
                if matched is None:
                    assigned[i] = i
                    primitives[i].basis_id = i
                    centers.append(i)
                else:
                    assigned[i] = matched
                    primitives[i].basis_id = matched
        return assigned

    @staticmethod
    def _embedding_distance(a: SitePrimitive, b: SitePrimitive) -> float:
        va = np.asarray(a.embedding, dtype=float)
        vb = np.asarray(b.embedding, dtype=float)
        return float(np.linalg.norm(va - vb))
