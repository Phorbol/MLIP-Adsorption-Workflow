from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from .primitives import SitePrimitive


@dataclass
class PrimitiveEmbeddingConfig:
    l2_distance_threshold: float = 0.30


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
            p.embedding = self._build_primitive_embedding(p, atom_features)
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
    def _build_primitive_embedding(primitive: SitePrimitive, atom_features: np.ndarray) -> np.ndarray:
        idx = np.array(primitive.atom_ids, dtype=int)
        feats = atom_features[idx]
        mean_feat = np.mean(feats, axis=0)
        return np.asarray(mean_feat, dtype=float)

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
