from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase import Atoms

from .mace_inference import MACEBatchInferencer


class DescriptorExtractor(Protocol):
    def transform(self, frames: list[Atoms]) -> np.ndarray:
        ...


@dataclass
class GeometryPairDistanceDescriptor:
    use_float64: bool = False

    def transform(self, frames: list[Atoms]) -> np.ndarray:
        if not frames:
            return np.empty((0, 0), dtype=float)
        n_atoms = len(frames[0])
        n_pairs = n_atoms * (n_atoms - 1) // 2
        dtype = np.float64 if self.use_float64 else np.float32
        feats = np.zeros((len(frames), n_pairs), dtype=dtype)
        for i, atoms in enumerate(frames):
            if len(atoms) != n_atoms:
                raise ValueError("All frames must have identical atom counts for descriptor extraction.")
            feats[i] = self._pair_distance_vector(atoms, dtype=dtype)
        return feats

    @staticmethod
    def _pair_distance_vector(atoms: Atoms, dtype: np.dtype) -> np.ndarray:
        n = len(atoms)
        out = np.empty(n * (n - 1) // 2, dtype=dtype)
        k = 0
        for a in range(n):
            for b in range(a + 1, n):
                out[k] = atoms.get_distance(a, b, mic=False)
                k += 1
        return out


@dataclass
class MACEInvariantDescriptor:
    inferencer: MACEBatchInferencer
    last_infer_metadata: dict | None = None

    def transform(self, frames: list[Atoms]) -> np.ndarray:
        out = self.inferencer.infer(frames)
        self.last_infer_metadata = dict(out.metadata)
        return np.asarray(out.descriptors, dtype=float)
