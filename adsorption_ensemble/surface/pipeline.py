from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from .classifier import SlabClassificationResult, SlabClassifier
from .detectors import ProbeScanDetector, SurfaceAtomDetector, SurfaceDetectionResult, VoxelFloodDetector
from .graph import ExposedSurfaceGraph, ExposedSurfaceGraphBuilder


@dataclass
class SurfaceContext:
    classification: SlabClassificationResult
    detection: SurfaceDetectionResult
    graph: ExposedSurfaceGraph


class SurfacePreprocessor:
    def __init__(
        self,
        classifier: SlabClassifier | None = None,
        primary_detector: SurfaceAtomDetector | None = None,
        fallback_detector: SurfaceAtomDetector | None = None,
        graph_builder: ExposedSurfaceGraphBuilder | None = None,
        min_surface_atoms: int = 6,
        prefer_upper_side: bool = True,
        side_split_gap: float = 1.5,
        target_surface_fraction: float | None = 0.25,
        target_count_mode: str = "fixed",
    ):
        self.classifier = classifier or SlabClassifier()
        self.primary_detector = primary_detector or ProbeScanDetector()
        self.fallback_detector = fallback_detector or VoxelFloodDetector()
        self.graph_builder = graph_builder or ExposedSurfaceGraphBuilder()
        self.min_surface_atoms = min_surface_atoms
        self.prefer_upper_side = prefer_upper_side
        self.side_split_gap = side_split_gap
        self.target_surface_fraction = target_surface_fraction
        self.target_count_mode = target_count_mode

    def build_context(self, slab: Atoms) -> SurfaceContext:
        cls = self.classifier.classify(slab)
        if not cls.is_slab or cls.normal_axis is None:
            raise ValueError("Input atoms are not classified as a slab.")
        primary = self.primary_detector.detect(slab, cls.normal_axis)
        fallback = self.fallback_detector.detect(slab, cls.normal_axis)
        detection = self._choose_detection(slab, primary, fallback)
        detection = self._select_adsorption_side(slab, detection, cls.normal_axis)
        detection = self._enforce_target_count(slab, detection, cls.normal_axis)
        graph = self.graph_builder.build(slab, detection.surface_atom_ids)
        return SurfaceContext(classification=cls, detection=detection, graph=graph)

    def _choose_detection(self, slab: Atoms, a: SurfaceDetectionResult, b: SurfaceDetectionResult) -> SurfaceDetectionResult:
        a_ok = self._is_detection_reasonable(slab, a)
        b_ok = self._is_detection_reasonable(slab, b)
        if a_ok:
            out = a
        elif b_ok:
            out = b
        else:
            out = a if len(a.surface_atom_ids) >= len(b.surface_atom_ids) else b
        diagnostics = dict(out.diagnostics)
        diagnostics["primary_method"] = a.diagnostics.get("method", "unknown")
        diagnostics["fallback_method"] = b.diagnostics.get("method", "unknown")
        diagnostics["primary_n"] = len(a.surface_atom_ids)
        diagnostics["fallback_n"] = len(b.surface_atom_ids)
        diagnostics["selected_detector"] = diagnostics.get("method", "unknown")
        return SurfaceDetectionResult(
            surface_atom_ids=out.surface_atom_ids,
            exposure_scores=out.exposure_scores,
            diagnostics=diagnostics,
        )

    def _is_detection_reasonable(self, slab: Atoms, detection: SurfaceDetectionResult) -> bool:
        n = len(detection.surface_atom_ids)
        n_atoms = len(slab)
        if n < self.min_surface_atoms:
            return False
        frac = n / max(1, n_atoms)
        if frac < 0.08 or frac > 0.70:
            return False
        return True

    def _select_adsorption_side(self, slab: Atoms, detection: SurfaceDetectionResult, normal_axis: int) -> SurfaceDetectionResult:
        if not self.prefer_upper_side or not detection.surface_atom_ids:
            return detection
        ids = detection.surface_atom_ids
        coords = slab.get_positions()[:, normal_axis]
        selected_coords = coords[ids]
        selected_scores = detection.exposure_scores[ids]
        levels = self._cluster_levels(selected_coords, selected_scores)
        if len(levels) < 2:
            return detection
        candidate = self._pick_upper_partition(levels, ids, coords)
        if candidate is None:
            diagnostics = dict(detection.diagnostics)
            diagnostics["side_filter"] = "skipped"
            return SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics)
        upper_ids, cutoff, max_gap = candidate
        if len(upper_ids) < self.min_surface_atoms:
            return detection
        new_scores = np.zeros_like(detection.exposure_scores)
        new_scores[upper_ids] = detection.exposure_scores[upper_ids]
        diagnostics = dict(detection.diagnostics)
        diagnostics["side_filter"] = "upper"
        diagnostics["side_split_gap"] = self.side_split_gap
        diagnostics["max_z_gap"] = max_gap
        diagnostics["surface_before_side_filter"] = len(ids)
        diagnostics["surface_after_side_filter"] = len(upper_ids)
        return SurfaceDetectionResult(surface_atom_ids=upper_ids, exposure_scores=new_scores, diagnostics=diagnostics)

    def _cluster_levels(self, z_values: np.ndarray, score_values: np.ndarray) -> list[tuple[float, float]]:
        if len(z_values) == 0:
            return []
        order = np.argsort(z_values)
        z_sorted = z_values[order]
        s_sorted = score_values[order]
        levels: list[list[float]] = []
        scores: list[list[float]] = []
        tol = 0.35
        for z, s in zip(z_sorted, s_sorted):
            if not levels or abs(z - levels[-1][-1]) > tol:
                levels.append([float(z)])
                scores.append([float(s)])
            else:
                levels[-1].append(float(z))
                scores[-1].append(float(s))
        out = []
        for lv, sc in zip(levels, scores):
            out.append((float(np.mean(lv)), float(np.mean(sc))))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def _pick_upper_partition(
        self,
        levels: list[tuple[float, float]],
        ids: list[int],
        coords: np.ndarray,
    ) -> tuple[list[int], float, float] | None:
        n_total = len(ids)
        best: tuple[float, list[int], float, float] | None = None
        for i in range(len(levels) - 1):
            z_hi, s_hi = levels[i]
            z_lo, s_lo = levels[i + 1]
            gap = z_hi - z_lo
            if gap < self.side_split_gap:
                continue
            cutoff = 0.5 * (z_hi + z_lo)
            upper_ids = [idx for idx in ids if coords[idx] >= cutoff]
            n_upper = len(upper_ids)
            if n_upper < self.min_surface_atoms or n_upper > int(0.8 * n_total):
                continue
            score = gap + max(0.0, s_hi - s_lo)
            if best is None or score > best[0]:
                best = (score, upper_ids, float(cutoff), float(gap))
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _enforce_target_count(self, slab: Atoms, detection: SurfaceDetectionResult, normal_axis: int) -> SurfaceDetectionResult:
        frac, n_layers_eff, mode = self._resolve_target_fraction(slab, normal_axis)
        if frac is None:
            return detection
        n_atoms = len(slab)
        modulo = max(2, int(n_layers_eff)) if n_layers_eff is not None else 4
        rem = n_atoms % modulo
        if rem == 0:
            target = int(round(n_atoms * frac))
            target_rule = "layer_exact"
        elif rem == 1:
            target = int(np.ceil(n_atoms * frac))
            target_rule = "layer_plus_defect"
        elif rem == modulo - 1:
            target = int(np.floor(n_atoms * frac))
            target_rule = "layer_minus_defect"
        else:
            target = int(round(n_atoms * frac))
            target_rule = "layer_rounded"
        if target <= 0:
            return detection
        ids = list(detection.surface_atom_ids)
        if len(ids) == target:
            diagnostics = dict(detection.diagnostics)
            diagnostics["target_surface_count"] = target
            diagnostics["surface_count_adjusted"] = False
            diagnostics["target_rule"] = target_rule
            diagnostics["target_fraction"] = frac
            diagnostics["target_mode"] = mode
            diagnostics["estimated_layers"] = n_layers_eff
            return SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics)
        scores = np.asarray(detection.exposure_scores, dtype=float)
        z = slab.get_positions()[:, normal_axis]
        rank = sorted(range(len(slab)), key=lambda i: (scores[i], z[i]), reverse=True)
        if len(ids) > target:
            in_set = set(ids)
            chosen = [i for i in rank if i in in_set][:target]
        else:
            chosen = ids[:]
            chosen_set = set(chosen)
            for i in rank:
                if i in chosen_set:
                    continue
                chosen.append(i)
                chosen_set.add(i)
                if len(chosen) >= target:
                    break
        new_scores = np.zeros_like(scores)
        new_scores[chosen] = scores[chosen]
        diagnostics = dict(detection.diagnostics)
        diagnostics["target_surface_count"] = target
        diagnostics["surface_count_before_target"] = len(ids)
        diagnostics["surface_count_after_target"] = len(chosen)
        diagnostics["surface_count_adjusted"] = True
        diagnostics["target_rule"] = target_rule
        diagnostics["target_fraction"] = frac
        diagnostics["target_mode"] = mode
        diagnostics["estimated_layers"] = n_layers_eff
        return SurfaceDetectionResult(surface_atom_ids=sorted(chosen), exposure_scores=new_scores, diagnostics=diagnostics)

    def _resolve_target_fraction(self, slab: Atoms, normal_axis: int) -> tuple[float | None, int | None, str]:
        if self.target_count_mode == "off":
            return None, None, "off"
        if self.target_count_mode == "auto_layers":
            n_layers = self._estimate_effective_layers(slab, normal_axis)
            if n_layers is None or n_layers < 2:
                if self.target_surface_fraction is None:
                    return None, None, "auto_layers_fallback_none"
                return float(self.target_surface_fraction), None, "auto_layers_fallback_fixed"
            return 1.0 / float(n_layers), int(n_layers), "auto_layers"
        if self.target_surface_fraction is None:
            return None, None, "fixed_none"
        return float(self.target_surface_fraction), None, "fixed"

    def _estimate_effective_layers(self, slab: Atoms, normal_axis: int) -> int | None:
        coords = slab.get_positions()[:, normal_axis]
        levels = self._cluster_coordinate_levels(coords, tol=0.35)
        if len(levels) < 2:
            return None
        counts = np.array([len(g) for g in levels], dtype=float)
        if counts.size == 0:
            return None
        core_ref = float(np.median(counts))
        keep_cut = max(2.0, 0.3 * core_ref)
        kept = [g for g in levels if len(g) >= keep_cut]
        if len(kept) >= 2:
            return len(kept)
        return len(levels)

    def _cluster_coordinate_levels(self, coords: np.ndarray, tol: float) -> list[list[float]]:
        if len(coords) == 0:
            return []
        out: list[list[float]] = []
        for val in sorted(float(x) for x in coords):
            if not out or abs(val - out[-1][-1]) > tol:
                out.append([val])
            else:
                out[-1].append(val)
        return out
