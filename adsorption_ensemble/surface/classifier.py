from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class SlabClassificationResult:
    is_slab: bool
    normal_axis: Optional[int]
    vacuum_lengths: tuple[float, float, float]
    confidence: float
    method: str


class SlabClassifier:
    def __init__(self, vacuum_threshold: float = 6.0, vacuum_ratio_threshold: float = 1.8):
        self.vacuum_threshold = vacuum_threshold
        self.vacuum_ratio_threshold = vacuum_ratio_threshold

    def classify(self, atoms: Atoms) -> SlabClassificationResult:
        if len(atoms) == 0:
            return SlabClassificationResult(False, None, (0.0, 0.0, 0.0), 0.0, "empty")
        result = self._classify_with_isolation(atoms)
        if result is not None:
            return result
        return self._classify_with_vacuum_heuristic(atoms)

    def _classify_with_isolation(self, atoms: Atoms) -> Optional[SlabClassificationResult]:
        try:
            from ase.geometry.dimensionality import analyze_dimensionality
        except Exception:
            return None
        try:
            intervals = analyze_dimensionality(atoms, method="RDA", merge=True)
        except Exception:
            return None
        if not intervals:
            return None
        best = intervals[0]
        dimtype = str(getattr(best, "dimtype", ""))
        score = float(getattr(best, "score", 0.0))
        cell_lengths = atoms.cell.lengths()
        wrapped = atoms.get_scaled_positions(wrap=True)
        span_frac = wrapped.max(axis=0) - wrapped.min(axis=0)
        span_frac = np.clip(span_frac, 0.0, 1.0)
        atomic_span = span_frac * cell_lengths
        vacuum = np.clip(cell_lengths - atomic_span, 0.0, None)
        axis = int(np.argmax(vacuum))
        if "2D" in dimtype and score >= 0.50:
            confidence = float(max(0.0, min(1.0, score)))
            return SlabClassificationResult(True, axis, tuple(float(x) for x in vacuum), confidence, "isolation_rda")
        if "3D" in dimtype and score >= 0.50:
            confidence = float(max(0.0, min(1.0, score)))
            return SlabClassificationResult(False, None, tuple(float(x) for x in vacuum), confidence, "isolation_rda")
        return None

    def _classify_with_vacuum_heuristic(self, atoms: Atoms) -> SlabClassificationResult:
        cell_lengths = atoms.cell.lengths()
        if np.any(cell_lengths < 1e-6):
            return SlabClassificationResult(False, None, tuple(float(x) for x in cell_lengths), 0.0, "no_cell")
        wrapped = atoms.get_scaled_positions(wrap=True)
        span_frac = wrapped.max(axis=0) - wrapped.min(axis=0)
        span_frac = np.clip(span_frac, 0.0, 1.0)
        atomic_span = span_frac * cell_lengths
        vacuum = cell_lengths - atomic_span
        vacuum = np.clip(vacuum, 0.0, None)
        axis = int(np.argmax(vacuum))
        sorted_vac = np.sort(vacuum)
        dominant = float(sorted_vac[-1])
        runner_up = float(sorted_vac[-2]) if len(sorted_vac) > 1 else 0.0
        ratio = (dominant + 1e-8) / (runner_up + 1e-8)
        is_slab = dominant >= self.vacuum_threshold and ratio >= self.vacuum_ratio_threshold
        confidence = 0.0
        if dominant > 0.0:
            confidence = float(min(1.0, 0.5 * (dominant / max(self.vacuum_threshold, 1e-6)) + 0.5 * min(2.0, ratio) / 2.0))
        return SlabClassificationResult(is_slab, axis if is_slab else None, tuple(float(x) for x in vacuum), confidence, "vacuum_heuristic")
