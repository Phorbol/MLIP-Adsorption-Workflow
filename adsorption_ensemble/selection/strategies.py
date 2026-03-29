from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


_FPS_NUMBA_SENTINEL = object()
_FPS_NUMBA_KERNEL = _FPS_NUMBA_SENTINEL


def _get_fps_numba_kernel():
    global _FPS_NUMBA_KERNEL
    if _FPS_NUMBA_KERNEL is None:
        return None
    if _FPS_NUMBA_KERNEL is not _FPS_NUMBA_SENTINEL:
        return _FPS_NUMBA_KERNEL
    try:
        import numba
    except Exception:
        _FPS_NUMBA_KERNEL = None
        return None

    @numba.njit(parallel=True, fastmath=True)
    def dist2_to_center(X: np.ndarray, center: np.ndarray) -> np.ndarray:
        n_points = X.shape[0]
        n_features = X.shape[1]
        out = np.empty(n_points, dtype=np.float32)
        for i in numba.prange(n_points):
            d = np.float32(0.0)
            for k in range(n_features):
                tmp = X[i, k] - center[k]
                d += tmp * tmp
            out[i] = d
        return out

    _FPS_NUMBA_KERNEL = dist2_to_center
    return dist2_to_center


@dataclass
class EnergyWindowFilter:
    delta_e: float

    def select(self, energies: np.ndarray, candidate_ids: list[int] | None = None) -> list[int]:
        if candidate_ids is None:
            candidate_ids = list(range(len(energies)))
        if not candidate_ids:
            return []
        subset = np.asarray([energies[i] for i in candidate_ids], dtype=float)
        e_min = float(np.min(subset))
        keep = [idx for idx in candidate_ids if energies[idx] <= e_min + self.delta_e]
        return keep


@dataclass
class RMSDSelector:
    threshold: float

    def select(self, features: np.ndarray, candidate_ids: list[int] | None = None) -> list[int]:
        if candidate_ids is None:
            candidate_ids = list(range(len(features)))
        selected: list[int] = []
        for idx in candidate_ids:
            if not selected:
                selected.append(idx)
                continue
            ref = features[selected]
            diff = ref - features[idx]
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            if float(np.min(dists)) >= self.threshold:
                selected.append(idx)
        return selected


@dataclass
class IterativeFPSResult:
    selected_ids: list[int]
    round_selected_ids: list[list[int]]
    stopped_by_convergence: bool
    metrics: dict = field(default_factory=dict)


@dataclass
class ConvergenceState:
    stop: bool
    metrics: dict = field(default_factory=dict)


class IterativeSelectionConvergence(Protocol):
    def reset(self) -> None:
        ...

    def update(self, newly_selected_ids: list[int], all_selected_ids: list[int], metadata_items: list[dict] | None) -> ConvergenceState:
        ...

    def summary(self) -> dict:
        ...


@dataclass
class SiteOccupancyConvergenceCriterion:
    bucket_keys: tuple[str, ...] = ("basis_id", "primitive_index", "conformer_id", "azimuth_index", "height_shift_index")
    min_new_bins: int = 0
    patience: int = 2
    min_rounds: int = 1
    _occupied_bins: set[tuple] = field(default_factory=set, init=False, repr=False)
    _stable_rounds: int = field(default=0, init=False, repr=False)
    _round_metrics: list[dict] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        self._occupied_bins = set()
        self._stable_rounds = 0
        self._round_metrics = []

    def _bucket_from_metadata(self, metadata: dict | None, fallback_id: int) -> tuple:
        if metadata is None:
            return ("selected_id", int(fallback_id))
        bucket: list[object] = []
        for key in self.bucket_keys:
            if key in metadata:
                bucket.append((key, metadata.get(key)))
        if not bucket:
            if "site_atom_ids" in metadata:
                bucket.append(("site_atom_ids", metadata.get("site_atom_ids")))
            else:
                bucket.append(("selected_id", int(fallback_id)))
        return tuple(bucket)

    def update(self, newly_selected_ids: list[int], all_selected_ids: list[int], metadata_items: list[dict] | None) -> ConvergenceState:
        old_count = len(self._occupied_bins)
        for pos, idx in enumerate(newly_selected_ids):
            meta = None if metadata_items is None or pos >= len(metadata_items) else metadata_items[pos]
            self._occupied_bins.add(self._bucket_from_metadata(meta, int(idx)))
        new_count = len(self._occupied_bins)
        new_bins = int(new_count - old_count)
        if new_bins <= int(self.min_new_bins):
            self._stable_rounds += 1
        else:
            self._stable_rounds = 0
        round_idx = len(self._round_metrics) + 1
        stop = bool(round_idx >= int(self.min_rounds) and self._stable_rounds >= int(self.patience))
        metrics = {
            "round_index": int(round_idx),
            "n_new_selected": int(len(newly_selected_ids)),
            "n_selected_total": int(len(all_selected_ids)),
            "n_occupied_bins": int(new_count),
            "n_new_bins": int(new_bins),
            "stable_rounds": int(self._stable_rounds),
            "stop": bool(stop),
        }
        self._round_metrics.append(metrics)
        return ConvergenceState(stop=stop, metrics=metrics)

    def summary(self) -> dict:
        return {
            "bucket_keys": list(self.bucket_keys),
            "min_new_bins": int(self.min_new_bins),
            "patience": int(self.patience),
            "min_rounds": int(self.min_rounds),
            "n_occupied_bins": int(len(self._occupied_bins)),
            "stable_rounds": int(self._stable_rounds),
            "round_metrics": [dict(x) for x in self._round_metrics],
        }


@dataclass
class FarthestPointSamplingSelector:
    random_seed: int = 0

    def select(
        self,
        features: np.ndarray,
        k: int,
        candidate_ids: list[int] | None = None,
        seed_ids: list[int] | None = None,
    ) -> list[int]:
        result = self.select_iterative(
            features=features,
            k=k,
            candidate_ids=candidate_ids,
            seed_ids=seed_ids,
        )
        return result.selected_ids

    def select_iterative(
        self,
        features: np.ndarray,
        k: int,
        candidate_ids: list[int] | None = None,
        seed_ids: list[int] | None = None,
        round_size: int | None = None,
        rounds: int | None = None,
        metadata_items: list[dict] | None = None,
        convergence: IterativeSelectionConvergence | None = None,
    ) -> IterativeFPSResult:
        if candidate_ids is None:
            candidate_ids = list(range(len(features)))
        if metadata_items is not None and len(metadata_items) != len(features):
            raise ValueError("metadata_items must align with features rows.")
        pool: list[int] = []
        seen_pool: set[int] = set()
        for idx in candidate_ids:
            ii = int(idx)
            if 0 <= ii < len(features) and ii not in seen_pool:
                pool.append(ii)
                seen_pool.add(ii)
        if not pool or k <= 0:
            return IterativeFPSResult(selected_ids=[], round_selected_ids=[], stopped_by_convergence=False, metrics={})
        if len(pool) <= k and not seed_ids and round_size is None and rounds is None:
            return IterativeFPSResult(
                selected_ids=list(pool),
                round_selected_ids=[list(pool)],
                stopped_by_convergence=False,
                metrics={"n_candidates": int(len(pool)), "n_selected": int(len(pool))},
            )
        rng = np.random.default_rng(self.random_seed)
        pool_arr = np.asarray(pool, dtype=int)
        kernel = _get_fps_numba_kernel()
        if kernel is None:
            pool_feats = np.asarray(features[pool_arr], dtype=float)
            min_dist2 = np.full(len(pool_arr), np.inf, dtype=float)
        else:
            pool_feats = np.asarray(features[pool_arr], dtype=np.float32)
            min_dist2 = np.full(len(pool_arr), np.inf, dtype=np.float32)
        n_pool = len(pool_arr)
        pool_lookup = {int(idx): int(pos) for pos, idx in enumerate(pool_arr.tolist())}
        chosen_pos = np.zeros(n_pool, dtype=bool)
        chosen: list[int] = []
        round_selected_ids: list[list[int]] = []
        if convergence is not None:
            convergence.reset()
        seed_clean: list[int] = []
        for idx in seed_ids or []:
            ii = int(idx)
            if ii in pool_lookup and ii not in seed_clean:
                seed_clean.append(ii)
        for seed in seed_clean[: max(0, int(k))]:
            pos = pool_lookup[seed]
            if chosen_pos[pos]:
                continue
            chosen_pos[pos] = True
            chosen.append(seed)
            last = pool_feats[pos]
            if kernel is None:
                diff = pool_feats - last
                dist2 = np.sum(diff * diff, axis=1)
            else:
                dist2 = kernel(pool_feats, last)
            min_dist2 = np.minimum(min_dist2, dist2)
            min_dist2[chosen_pos] = -1.0
        if not chosen and len(chosen) < k:
            start_pos = int(rng.integers(0, n_pool))
            chosen_pos[start_pos] = True
            chosen.append(int(pool_arr[start_pos]))
            last = pool_feats[start_pos]
            if kernel is None:
                diff = pool_feats - last
                dist2 = np.sum(diff * diff, axis=1)
            else:
                dist2 = kernel(pool_feats, last)
            min_dist2 = np.minimum(min_dist2, dist2)
            min_dist2[chosen_pos] = -1.0
        added_since_round = 0
        round_new_ids: list[int] = []
        max_rounds = max(1, int(rounds)) if rounds is not None else None
        effective_round_size = max(1, int(round_size)) if round_size is not None else None
        stopped_by_convergence = False
        while len(chosen) < min(int(k), n_pool):
            start_pos = int(np.argmax(min_dist2))
            if chosen_pos[start_pos]:
                break
            chosen_pos[start_pos] = True
            chosen.append(int(pool_arr[start_pos]))
            round_new_ids.append(int(pool_arr[start_pos]))
            added_since_round += 1
            last = pool_feats[start_pos]
            if kernel is None:
                diff = pool_feats - last
                dist2 = np.sum(diff * diff, axis=1)
            else:
                dist2 = kernel(pool_feats, last)
            min_dist2 = np.minimum(min_dist2, dist2)
            min_dist2[chosen_pos] = -1.0
            should_close_round = False
            if effective_round_size is None:
                should_close_round = len(chosen) >= min(int(k), n_pool)
            elif added_since_round >= effective_round_size:
                should_close_round = True
            if should_close_round:
                if round_new_ids:
                    round_selected_ids.append(list(round_new_ids))
                    if convergence is not None:
                        metadata_new = None if metadata_items is None else [metadata_items[i] for i in round_new_ids]
                        conv_state = convergence.update(
                            newly_selected_ids=list(round_new_ids),
                            all_selected_ids=list(chosen),
                            metadata_items=metadata_new,
                        )
                        if conv_state.stop:
                            stopped_by_convergence = True
                            break
                    if max_rounds is not None and len(round_selected_ids) >= max_rounds:
                        break
                round_new_ids = []
                added_since_round = 0
        if round_new_ids and not stopped_by_convergence:
            round_selected_ids.append(list(round_new_ids))
            if convergence is not None:
                metadata_new = None if metadata_items is None else [metadata_items[i] for i in round_new_ids]
                conv_state = convergence.update(
                    newly_selected_ids=list(round_new_ids),
                    all_selected_ids=list(chosen),
                    metadata_items=metadata_new,
                )
                stopped_by_convergence = bool(conv_state.stop)
        metrics = {
            "n_candidates": int(n_pool),
            "n_selected": int(len(chosen)),
            "n_seed": int(len(seed_clean)),
            "n_rounds": int(len(round_selected_ids)),
        }
        if convergence is not None:
            metrics["convergence"] = convergence.summary()
        return IterativeFPSResult(
            selected_ids=chosen[: min(int(k), len(chosen))],
            round_selected_ids=round_selected_ids,
            stopped_by_convergence=bool(stopped_by_convergence),
            metrics=metrics,
        )


@dataclass
class DualThresholdSelector:
    energy_window: EnergyWindowFilter
    rmsd_selector: RMSDSelector

    def select(self, energies: np.ndarray, features: np.ndarray, candidate_ids: list[int] | None = None) -> list[int]:
        kept = self.energy_window.select(energies=energies, candidate_ids=candidate_ids)
        ordered = sorted(kept, key=lambda i: float(energies[i]))
        return self.rmsd_selector.select(features=features, candidate_ids=ordered)
