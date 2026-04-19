from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import build_neighbor_list, natural_cutoffs

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
        level_z_tolerance: float = 0.35,
        min_exposure_level_score: float = 0.10,
        relative_exposure_level_score: float = 0.10,
        rescue_subsurface_species: bool = True,
        rescue_anchor_symbols: tuple[str, ...] = ("O", "N", "F", "Cl", "Br", "I", "S"),
        rescue_subsurface_depth: float = 2.5,
        rescue_level_tolerance: float = 0.35,
        rescue_stepped_surface_by_coordination: bool = True,
        coordination_rescue_depth: float = 2.2,
        coordination_rescue_deficit: int = 2,
        coordination_rescue_neighbor_scale: float = 1.2,
        adaptive_target_relative_level_score: float = 0.25,
        adaptive_target_absolute_level_score: float = 0.20,
        adaptive_target_skip_multispecies: bool = True,
        adaptive_target_skip_coordination_rescue: bool = True,
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
        self.level_z_tolerance = level_z_tolerance
        self.min_exposure_level_score = min_exposure_level_score
        self.relative_exposure_level_score = relative_exposure_level_score
        self.rescue_subsurface_species = rescue_subsurface_species
        self.rescue_anchor_symbols = tuple(str(x) for x in rescue_anchor_symbols)
        self.rescue_subsurface_depth = rescue_subsurface_depth
        self.rescue_level_tolerance = rescue_level_tolerance
        self.rescue_stepped_surface_by_coordination = bool(rescue_stepped_surface_by_coordination)
        self.coordination_rescue_depth = coordination_rescue_depth
        self.coordination_rescue_deficit = coordination_rescue_deficit
        self.coordination_rescue_neighbor_scale = coordination_rescue_neighbor_scale
        self.adaptive_target_relative_level_score = float(adaptive_target_relative_level_score)
        self.adaptive_target_absolute_level_score = float(adaptive_target_absolute_level_score)
        self.adaptive_target_skip_multispecies = bool(adaptive_target_skip_multispecies)
        self.adaptive_target_skip_coordination_rescue = bool(adaptive_target_skip_coordination_rescue)

    def build_context(self, slab: Atoms) -> SurfaceContext:
        cls = self.classifier.classify(slab)
        if not cls.is_slab or cls.normal_axis is None:
            raise ValueError("Input atoms are not classified as a slab.")
        primary = self.primary_detector.detect(slab, cls.normal_axis)
        fallback = self.fallback_detector.detect(slab, cls.normal_axis)
        detection = self._choose_detection(slab, primary, fallback)
        detection = self._select_adsorption_side(slab, detection, cls.normal_axis)
        detection = self._prune_low_exposure_levels(slab, detection, cls.normal_axis)
        detection = self._rescue_subsurface_multicomponent_sites(slab, detection, cls.normal_axis)
        detection = self._rescue_stepped_surface_levels(slab, detection, cls.normal_axis)
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
        for z, s in zip(z_sorted, s_sorted):
            if not levels or abs(z - levels[-1][-1]) > self.level_z_tolerance:
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

    def _prune_low_exposure_levels(self, slab: Atoms, detection: SurfaceDetectionResult, normal_axis: int) -> SurfaceDetectionResult:
        ids = list(detection.surface_atom_ids)
        if len(ids) <= 1:
            return detection
        level_groups = self._group_level_ids(slab, ids, normal_axis)
        if len(level_groups) <= 1:
            diagnostics = dict(detection.diagnostics)
            diagnostics["exposure_level_filter"] = "single_level"
            return SurfaceDetectionResult(
                surface_atom_ids=ids,
                exposure_scores=detection.exposure_scores,
                diagnostics=diagnostics,
            )
        scores = np.asarray(detection.exposure_scores, dtype=float)
        level_stats: list[tuple[float, float, list[int]]] = []
        for group in level_groups:
            z_mean = float(np.mean(slab.get_positions()[group, normal_axis]))
            score_mean = float(np.mean(scores[group]))
            level_stats.append((z_mean, score_mean, group))
        max_level_score = max(score for _, score, _ in level_stats)
        score_cutoff = max(self.min_exposure_level_score, self.relative_exposure_level_score * max_level_score)
        kept_groups = [group for _, score, group in level_stats if score >= score_cutoff]
        if not kept_groups:
            kept_groups = [max(level_stats, key=lambda item: item[1])[2]]
        kept_ids = sorted(int(i) for group in kept_groups for i in group)
        diagnostics = dict(detection.diagnostics)
        diagnostics["exposure_level_cutoff"] = float(score_cutoff)
        diagnostics["surface_before_exposure_filter"] = len(ids)
        diagnostics["surface_after_exposure_filter"] = len(kept_ids)
        diagnostics["exposure_level_scores"] = [
            {"z_mean": float(z_mean), "mean_score": float(score), "count": int(len(group))}
            for z_mean, score, group in level_stats
        ]
        if len(kept_ids) == len(ids):
            diagnostics["exposure_level_filter"] = "unchanged"
            return SurfaceDetectionResult(
                surface_atom_ids=ids,
                exposure_scores=detection.exposure_scores,
                diagnostics=diagnostics,
            )
        diagnostics["exposure_level_filter"] = "applied"
        new_scores = np.zeros_like(scores)
        new_scores[kept_ids] = scores[kept_ids]
        return SurfaceDetectionResult(surface_atom_ids=kept_ids, exposure_scores=new_scores, diagnostics=diagnostics)

    def _group_level_ids(self, slab: Atoms, ids: list[int], normal_axis: int) -> list[list[int]]:
        coords = slab.get_positions()[:, normal_axis]
        ordered = sorted(((float(coords[i]), int(i)) for i in ids), key=lambda item: item[0], reverse=True)
        groups: list[list[int]] = []
        anchors: list[float] = []
        for z_val, idx in ordered:
            if not groups or abs(z_val - anchors[-1]) > self.level_z_tolerance:
                groups.append([idx])
                anchors.append(z_val)
            else:
                groups[-1].append(idx)
        return groups

    def _enforce_target_count(self, slab: Atoms, detection: SurfaceDetectionResult, normal_axis: int) -> SurfaceDetectionResult:
        if self.target_count_mode == "adaptive":
            detection, use_target = self._apply_adaptive_target_gate(slab, detection, normal_axis)
            if not bool(use_target):
                return detection
            if self.target_surface_fraction is None:
                diagnostics = dict(detection.diagnostics)
                diagnostics["target_mode"] = "adaptive_no_fraction"
                diagnostics["surface_count_adjusted"] = False
                return SurfaceDetectionResult(
                    surface_atom_ids=list(detection.surface_atom_ids),
                    exposure_scores=detection.exposure_scores,
                    diagnostics=diagnostics,
                )
            frac = float(self.target_surface_fraction)
            n_layers_eff = None
            mode = "adaptive"
        else:
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
        if len(ids) < target:
            diagnostics = dict(detection.diagnostics)
            diagnostics["target_surface_count"] = target
            diagnostics["surface_count_before_target"] = len(ids)
            diagnostics["surface_count_after_target"] = len(ids)
            diagnostics["surface_count_adjusted"] = False
            diagnostics["target_count_upsample_skipped"] = True
            diagnostics["target_rule"] = target_rule
            diagnostics["target_fraction"] = frac
            diagnostics["target_mode"] = mode
            diagnostics["estimated_layers"] = n_layers_eff
            return SurfaceDetectionResult(
                surface_atom_ids=ids,
                exposure_scores=detection.exposure_scores,
                diagnostics=diagnostics,
            )
        scores = np.asarray(detection.exposure_scores, dtype=float)
        z = slab.get_positions()[:, normal_axis]
        rank = sorted(range(len(slab)), key=lambda i: (scores[i], z[i]), reverse=True)
        in_set = set(ids)
        chosen = [i for i in rank if i in in_set][:target]
        new_scores = np.zeros_like(scores)
        new_scores[chosen] = scores[chosen]
        diagnostics = dict(detection.diagnostics)
        diagnostics["target_surface_count"] = target
        diagnostics["surface_count_before_target"] = len(ids)
        diagnostics["surface_count_after_target"] = len(chosen)
        diagnostics["surface_count_adjusted"] = True
        diagnostics["target_count_upsample_skipped"] = False
        diagnostics["target_rule"] = target_rule
        diagnostics["target_fraction"] = frac
        diagnostics["target_mode"] = mode
        diagnostics["estimated_layers"] = n_layers_eff
        return SurfaceDetectionResult(surface_atom_ids=sorted(chosen), exposure_scores=new_scores, diagnostics=diagnostics)

    def _apply_adaptive_target_gate(
        self,
        slab: Atoms,
        detection: SurfaceDetectionResult,
        normal_axis: int,
    ) -> tuple[SurfaceDetectionResult, bool]:
        ids = sorted(int(i) for i in detection.surface_atom_ids)
        diagnostics = dict(detection.diagnostics)
        diagnostics["adaptive_target_relative_level_score"] = float(self.adaptive_target_relative_level_score)
        diagnostics["adaptive_target_absolute_level_score"] = float(self.adaptive_target_absolute_level_score)
        if not ids:
            diagnostics["adaptive_target_decision"] = "skip_empty"
            return (
                SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
                False,
            )
        if bool(self.adaptive_target_skip_coordination_rescue) and str(diagnostics.get("coordination_rescue")) == "applied":
            diagnostics["adaptive_target_decision"] = "skip_coordination_rescue"
            return (
                SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
                False,
            )
        surface_symbols = sorted({str(slab[int(i)].symbol) for i in ids})
        if bool(self.adaptive_target_skip_multispecies) and len(surface_symbols) > 1:
            diagnostics["adaptive_target_decision"] = "skip_surface_multispecies"
            return (
                SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
                False,
            )
        if bool(self.adaptive_target_skip_multispecies) and str(diagnostics.get("subsurface_species_rescue")) == "applied":
            diagnostics["adaptive_target_decision"] = "skip_subsurface_multispecies"
            return (
                SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
                False,
            )
        level_scores_raw = diagnostics.get("exposure_level_scores")
        if isinstance(level_scores_raw, list) and level_scores_raw:
            level_scores = [dict(x) for x in level_scores_raw]
        else:
            level_groups = self._group_level_ids(slab, ids, normal_axis)
            score_arr = np.asarray(detection.exposure_scores, dtype=float)
            level_scores = []
            for group in level_groups:
                z_mean = float(np.mean(slab.get_positions()[group, normal_axis]))
                mean_score = float(np.mean(score_arr[group])) if len(group) > 0 else 0.0
                level_scores.append({"z_mean": z_mean, "mean_score": mean_score, "count": int(len(group))})
            level_scores.sort(key=lambda row: float(row.get("z_mean", 0.0)), reverse=True)
        diagnostics["adaptive_target_level_scores"] = [
            {
                "z_mean": float(row.get("z_mean", 0.0)),
                "mean_score": float(row.get("mean_score", 0.0)),
                "count": int(row.get("count", 0)),
            }
            for row in level_scores
        ]
        if len(level_scores) <= 1:
            diagnostics["adaptive_target_decision"] = "apply_single_level"
            return (
                SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
                True,
            )
        top_score = float(level_scores[0].get("mean_score", 0.0))
        second_score = float(level_scores[1].get("mean_score", 0.0))
        cutoff = max(
            float(self.adaptive_target_absolute_level_score),
            float(self.adaptive_target_relative_level_score) * max(top_score, 0.0),
        )
        diagnostics["adaptive_target_second_level_score"] = float(second_score)
        diagnostics["adaptive_target_score_cutoff"] = float(cutoff)
        if second_score <= cutoff:
            diagnostics["adaptive_target_decision"] = "apply_weak_second_level"
            return (
                SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
                True,
            )
        diagnostics["adaptive_target_decision"] = "keep_multilevel"
        return (
            SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics),
            False,
        )

    def _rescue_subsurface_multicomponent_sites(
        self,
        slab: Atoms,
        detection: SurfaceDetectionResult,
        normal_axis: int,
    ) -> SurfaceDetectionResult:
        ids = sorted(int(i) for i in detection.surface_atom_ids)
        if not bool(self.rescue_subsurface_species) or not ids:
            return detection
        symbols = slab.get_chemical_symbols()
        all_species = sorted(set(str(s) for s in symbols))
        if len(all_species) <= 1:
            return detection
        surface_species = sorted({str(symbols[i]) for i in ids})
        if len(surface_species) != 1:
            diagnostics = dict(detection.diagnostics)
            diagnostics["subsurface_species_rescue"] = "skipped_surface_multispecies"
            return SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics)
        anchor_symbol = str(surface_species[0])
        if anchor_symbol not in set(self.rescue_anchor_symbols):
            diagnostics = dict(detection.diagnostics)
            diagnostics["subsurface_species_rescue"] = "skipped_anchor_symbol"
            return SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics)
        coords = np.asarray(slab.get_positions()[:, normal_axis], dtype=float)
        top_surface_z = float(np.max(coords[ids]))
        extras: list[int] = []
        rescued_species: list[str] = []
        for species in all_species:
            if species == anchor_symbol:
                continue
            cand = [i for i, sym in enumerate(symbols) if str(sym) == species and i not in ids]
            if not cand:
                continue
            top_species_z = float(np.max(coords[cand]))
            depth = float(top_surface_z - top_species_z)
            if depth < -1e-8 or depth > float(self.rescue_subsurface_depth):
                continue
            keep = [
                int(i)
                for i in cand
                if abs(float(coords[i]) - top_species_z) <= float(self.rescue_level_tolerance)
            ]
            if keep:
                extras.extend(keep)
                rescued_species.append(str(species))
        if not extras:
            diagnostics = dict(detection.diagnostics)
            diagnostics["subsurface_species_rescue"] = "no_candidates"
            return SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics)
        merged = sorted(set(ids).union(int(i) for i in extras))
        scores = np.asarray(detection.exposure_scores, dtype=float)
        new_scores = np.asarray(scores, dtype=float).copy()
        anchor_score = float(np.max(scores[ids])) if ids else 1.0
        for idx in extras:
            depth = max(0.0, top_surface_z - float(coords[idx]))
            depth_factor = max(0.2, 1.0 - depth / max(float(self.rescue_subsurface_depth), 1.0e-8))
            new_scores[int(idx)] = max(new_scores[int(idx)], anchor_score * depth_factor)
        diagnostics = dict(detection.diagnostics)
        diagnostics["subsurface_species_rescue"] = "applied"
        diagnostics["surface_before_subsurface_rescue"] = int(len(ids))
        diagnostics["surface_after_subsurface_rescue"] = int(len(merged))
        diagnostics["subsurface_rescue_added_n"] = int(len(set(extras)))
        diagnostics["subsurface_rescue_anchor_symbol"] = str(anchor_symbol)
        diagnostics["subsurface_rescue_species"] = [str(x) for x in sorted(set(rescued_species))]
        diagnostics["subsurface_rescue_depth"] = float(self.rescue_subsurface_depth)
        diagnostics["subsurface_rescue_level_tolerance"] = float(self.rescue_level_tolerance)
        return SurfaceDetectionResult(surface_atom_ids=merged, exposure_scores=new_scores, diagnostics=diagnostics)

    def _rescue_stepped_surface_levels(
        self,
        slab: Atoms,
        detection: SurfaceDetectionResult,
        normal_axis: int,
    ) -> SurfaceDetectionResult:
        ids = sorted(int(i) for i in detection.surface_atom_ids)
        if not bool(self.rescue_stepped_surface_by_coordination) or not ids:
            return detection
        coords = np.asarray(slab.get_positions()[:, normal_axis], dtype=float)
        top_surface_z = float(np.max(coords[ids]))
        cutoffs = natural_cutoffs(slab, mult=float(self.coordination_rescue_neighbor_scale))
        nl = build_neighbor_list(slab, cutoffs=cutoffs, bothways=True, self_interaction=False, skin=0.0)
        coordination = np.asarray([len(nl.get_neighbors(i)[0]) for i in range(len(slab))], dtype=int)
        symbols = slab.get_chemical_symbols()
        species_ref: dict[str, int] = {}
        for sym in sorted(set(str(s) for s in symbols)):
            idx = [i for i, s in enumerate(symbols) if str(s) == sym]
            if idx:
                species_ref[str(sym)] = int(np.max(coordination[idx]))
        extras: list[int] = []
        rescue_rows: list[dict[str, float | int | str]] = []
        for i in range(len(slab)):
            if i in ids:
                continue
            depth = float(top_surface_z - coords[i])
            if depth < 1.0e-8 or depth > float(self.coordination_rescue_depth):
                continue
            sym = str(symbols[i])
            ref = int(species_ref.get(sym, int(np.max(coordination))))
            deficit = int(ref - coordination[i])
            if deficit < int(self.coordination_rescue_deficit):
                continue
            extras.append(int(i))
            rescue_rows.append(
                {
                    "atom_id": int(i),
                    "symbol": sym,
                    "z": float(coords[i]),
                    "depth": float(depth),
                    "coordination": int(coordination[i]),
                    "coordination_reference": int(ref),
                    "coordination_deficit": int(deficit),
                }
            )
        diagnostics = dict(detection.diagnostics)
        diagnostics["coordination_rescue_enabled"] = True
        diagnostics["coordination_rescue_depth"] = float(self.coordination_rescue_depth)
        diagnostics["coordination_rescue_deficit"] = int(self.coordination_rescue_deficit)
        diagnostics["coordination_rescue_neighbor_scale"] = float(self.coordination_rescue_neighbor_scale)
        diagnostics["coordination_species_reference"] = {str(k): int(v) for k, v in species_ref.items()}
        if not extras:
            diagnostics["coordination_rescue"] = "no_candidates"
            return SurfaceDetectionResult(surface_atom_ids=ids, exposure_scores=detection.exposure_scores, diagnostics=diagnostics)
        merged = sorted(set(ids).union(int(i) for i in extras))
        scores = np.asarray(detection.exposure_scores, dtype=float)
        new_scores = np.asarray(scores, dtype=float).copy()
        anchor_score = float(np.max(scores[ids])) if ids else 1.0
        for row in rescue_rows:
            idx = int(row["atom_id"])
            deficit = float(row["coordination_deficit"])
            depth = float(row["depth"])
            deficit_factor = max(0.25, min(1.0, deficit / max(1.0, float(self.coordination_rescue_deficit))))
            depth_factor = max(0.2, 1.0 - depth / max(float(self.coordination_rescue_depth), 1.0e-8))
            new_scores[idx] = max(new_scores[idx], anchor_score * max(deficit_factor, depth_factor))
        diagnostics["coordination_rescue"] = "applied"
        diagnostics["surface_before_coordination_rescue"] = int(len(ids))
        diagnostics["surface_after_coordination_rescue"] = int(len(merged))
        diagnostics["coordination_rescue_rows"] = rescue_rows
        return SurfaceDetectionResult(surface_atom_ids=merged, exposure_scores=new_scores, diagnostics=diagnostics)

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
