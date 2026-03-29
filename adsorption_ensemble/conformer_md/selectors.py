from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adsorption_ensemble.selection import (
    DualThresholdSelector,
    EnergyWindowFilter,
    FarthestPointSamplingSelector,
    PCAGridOccupancyConvergenceCriterion,
    RMSDSelector,
)

from .config import SelectionConfig


@dataclass
class SelectionResult:
    preselected_ids: list[int]
    final_ids: list[int]


class ConformerSelector:
    def __init__(self, config: SelectionConfig):
        self.config = config
        self._fps = FarthestPointSamplingSelector(random_seed=config.fps_seed)
        self._dual = DualThresholdSelector(
            energy_window=EnergyWindowFilter(delta_e=config.energy_window_ev),
            rmsd_selector=RMSDSelector(threshold=config.rmsd_threshold),
        )

    def select(self, energies: np.ndarray, features: np.ndarray) -> SelectionResult:
        if len(energies) != len(features):
            raise ValueError("Energies and features must have identical frame counts.")
        pre = self._preselect(energies=energies, features=features)
        final_ids = self.dual_filter(energies=energies, features=features, candidate_ids=pre)
        return SelectionResult(preselected_ids=pre, final_ids=final_ids)

    def dual_filter(self, energies: np.ndarray, features: np.ndarray, candidate_ids: list[int] | None = None) -> list[int]:
        return self._dual.select(energies=energies, features=features, candidate_ids=candidate_ids)

    def _preselect(self, energies: np.ndarray, features: np.ndarray) -> list[int]:
        if len(features) == 0:
            return []
        k = max(1, min(self.config.preselect_k, len(features)))
        mode = self.config.mode.lower()
        if mode == "fps":
            convergence = None
            if bool(self.config.fps_convergence_enable):
                convergence = PCAGridOccupancyConvergenceCriterion(
                    features=np.asarray(features, dtype=float),
                    pca_variance_threshold=float(self.config.fps_convergence_pca_var),
                    grid_bins=int(self.config.fps_convergence_grid_bins),
                    min_rounds=int(self.config.fps_convergence_min_rounds),
                    patience=int(self.config.fps_convergence_patience),
                    min_coverage_gain=float(self.config.fps_convergence_min_coverage_gain),
                    min_novelty=float(self.config.fps_convergence_min_novelty),
                )
            result = self._fps.select_iterative(
                features=features,
                k=k,
                seed_ids=list(self.config.fps_seed_indices),
                round_size=self.config.fps_round_size,
                rounds=self.config.fps_rounds,
                convergence=convergence,
            )
            return result.selected_ids
        if mode == "fps_pca_kmeans":
            return self._fps_pca_kmeans_preselect(energies=energies, features=features, k=k, seed=self.config.fps_seed)
        if mode == "kmeans":
            return self._kmeans_pick_low_energy(energies=energies, features=features, k=k, seed=self.config.fps_seed)
        raise ValueError(f"Unsupported selection mode: {self.config.mode}")

    def _fps_pca_kmeans_preselect(self, energies: np.ndarray, features: np.ndarray, k: int, seed: int) -> list[int]:
        n = len(features)
        if n <= k:
            return list(range(n))
        pool_k = int(max(k, min(n, k * max(1, self.config.fps_pool_factor))))
        fps_ids = self._fps.select(
            features=features,
            k=pool_k,
            seed_ids=list(self.config.fps_seed_indices),
        )
        sub_feats = np.asarray([features[i] for i in fps_ids], dtype=float)
        reduced = self._pca_keep_variance(sub_feats, threshold=self.config.pca_variance_threshold)
        sub_energies = np.asarray([energies[i] for i in fps_ids], dtype=float)
        local_pick = self._kmeans_pick_low_energy(
            energies=sub_energies,
            features=reduced,
            k=min(k, len(fps_ids)),
            seed=seed,
        )
        return [int(fps_ids[i]) for i in local_pick]

    @staticmethod
    def _kmeans_pick_low_energy(energies: np.ndarray, features: np.ndarray, k: int, seed: int) -> list[int]:
        if len(features) <= k:
            return list(range(len(features)))
        reduced = ConformerSelector._pca(features, n_components=min(8, features.shape[1]))
        labels = ConformerSelector._kmeans_labels(reduced, n_clusters=k, seed=seed)
        picked: list[int] = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            best = int(members[np.argmin(energies[members])])
            picked.append(best)
        if not picked:
            return list(range(min(k, len(features))))
        picked = sorted(set(picked))
        return picked

    @staticmethod
    def _pca(features: np.ndarray, n_components: int) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        x = x - np.mean(x, axis=0, keepdims=True)
        if x.shape[1] == 0:
            return np.zeros((x.shape[0], 1), dtype=float)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        comp = vt[: max(1, n_components)]
        return x @ comp.T

    @staticmethod
    def _pca_keep_variance(features: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        x = x - np.mean(x, axis=0, keepdims=True)
        if x.shape[1] == 0:
            return np.zeros((x.shape[0], 1), dtype=float)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        var = s * s
        if np.sum(var) <= 0:
            comp = vt[:1]
            return x @ comp.T
        ratio = np.cumsum(var) / np.sum(var)
        n_comp = int(np.searchsorted(ratio, float(threshold), side="left") + 1)
        n_comp = max(1, min(n_comp, vt.shape[0]))
        comp = vt[:n_comp]
        return x @ comp.T

    @staticmethod
    def _kmeans_labels(features: np.ndarray, n_clusters: int, seed: int, n_iter: int = 30) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = len(features)
        if n_clusters >= n:
            return np.arange(n, dtype=int)
        init_ids = rng.choice(n, size=n_clusters, replace=False)
        centers = features[init_ids].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(n_iter):
            d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for i in range(n_clusters):
                members = features[labels == i]
                if len(members) == 0:
                    centers[i] = features[int(rng.integers(0, n))]
                else:
                    centers[i] = np.mean(members, axis=0)
        return labels
