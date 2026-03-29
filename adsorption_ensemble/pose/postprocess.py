from __future__ import annotations

from pathlib import Path

import numpy as np

from adsorption_ensemble.selection import FarthestPointSamplingSelector, SiteOccupancyConvergenceCriterion


def _normalize_seed_indices(seed_indices: tuple[int, ...] | list[int] | None) -> list[int]:
    if not seed_indices:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for idx in seed_indices:
        ii = int(idx)
        if ii >= 0 and ii not in seen:
            out.append(ii)
            seen.add(ii)
    return out


def run_iterative_pose_fps_preselection(
    case_out: Path,
    features: np.ndarray,
    pooled,
    *,
    random_seed: int,
    k: int,
    seed_indices: tuple[int, ...] | list[int] | None = None,
    round_size: int | None = None,
    rounds: int | None = None,
    occupancy_convergence: bool = False,
    occupancy_min_new_bins: int = 0,
    occupancy_patience: int = 2,
    occupancy_min_rounds: int = 1,
) -> dict:
    fps = FarthestPointSamplingSelector(random_seed=int(random_seed))
    metadata_items = [dict(getattr(a, "info", {}) or {}) for a in pooled]
    convergence = None
    if occupancy_convergence:
        convergence = SiteOccupancyConvergenceCriterion(
            min_new_bins=int(occupancy_min_new_bins),
            patience=int(occupancy_patience),
            min_rounds=int(occupancy_min_rounds),
        )
    result = fps.select_iterative(
        features=np.asarray(features, dtype=float),
        k=int(k),
        seed_ids=_normalize_seed_indices(seed_indices),
        round_size=(None if round_size is None else int(round_size)),
        rounds=(None if rounds is None else int(rounds)),
        metadata_items=metadata_items,
        convergence=convergence,
    )
    round_dir = case_out / "fps_rounds"
    round_dir.mkdir(parents=True, exist_ok=True)
    cumulative: list[int] = []
    for idx, round_ids in enumerate(result.round_selected_ids, start=1):
        np.save(round_dir / f"round_{idx:03d}_indices.npy", np.asarray(round_ids, dtype=int))
        cumulative.extend(int(x) for x in round_ids)
        np.save(round_dir / f"round_{idx:03d}_cumulative_indices.npy", np.asarray(cumulative, dtype=int))
    metrics = dict(result.metrics)
    metrics.update(
        {
            "selected_ids": [int(x) for x in result.selected_ids],
            "round_dir": round_dir.as_posix(),
            "stopped_by_convergence": bool(result.stopped_by_convergence),
        }
    )
    return {
        "selected_ids": [int(x) for x in result.selected_ids],
        "round_dir": round_dir,
        "metrics": metrics,
    }
