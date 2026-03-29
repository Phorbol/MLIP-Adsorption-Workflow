import unittest

import numpy as np
from ase.build import molecule

from adsorption_ensemble.pose.postprocess import run_iterative_pose_fps_preselection
from adsorption_ensemble.selection.strategies import (
    FarthestPointSamplingSelector,
    PCAGridOccupancyConvergenceCriterion,
)


class TestFarthestPointSamplingSelector(unittest.TestCase):
    def test_returns_k_unique_indices(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(200, 8)).astype(float)
        sel = FarthestPointSamplingSelector(random_seed=1)
        out = sel.select(features=features, k=32)
        self.assertEqual(len(out), 32)
        self.assertEqual(len(set(out)), 32)
        self.assertTrue(all(0 <= i < len(features) for i in out))

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(123)
        features = rng.normal(size=(128, 12)).astype(float)
        out1 = FarthestPointSamplingSelector(random_seed=7).select(features=features, k=25)
        out2 = FarthestPointSamplingSelector(random_seed=7).select(features=features, k=25)
        self.assertEqual(out1, out2)

    def test_degenerate_features_still_selects_k(self):
        features = np.zeros((50, 10), dtype=float)
        out = FarthestPointSamplingSelector(random_seed=0).select(features=features, k=30)
        self.assertEqual(len(out), 30)
        self.assertEqual(len(set(out)), 30)

    def test_pca_grid_convergence_stops_iterative_fps_early(self):
        corners = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        features = np.repeat(corners, repeats=8, axis=0)
        convergence = PCAGridOccupancyConvergenceCriterion(
            features=features,
            pca_variance_threshold=0.95,
            grid_bins=2,
            min_rounds=2,
            patience=1,
            min_coverage_gain=1e-6,
            min_novelty=0.25,
        )
        result = FarthestPointSamplingSelector(random_seed=0).select_iterative(
            features=features,
            k=16,
            round_size=4,
            rounds=10,
            convergence=convergence,
        )
        self.assertTrue(result.stopped_by_convergence)
        self.assertLess(len(result.round_selected_ids), 10)
        self.assertIn("convergence", result.metrics)
        last_round = result.metrics["convergence"]["round_metrics"][-1]
        self.assertIn("coverage", last_round)
        self.assertIn("novelty", last_round)

    def test_pose_preselection_accepts_grid_convergence(self):
        features = np.repeat(np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float), repeats=6, axis=0)
        pooled = []
        for idx in range(len(features)):
            atoms = molecule("H2O")
            atoms.info["basis_id"] = idx % 2
            atoms.info["primitive_index"] = idx
            pooled.append(atoms)
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            out = run_iterative_pose_fps_preselection(
                case_out=Path(td),
                features=features,
                pooled=pooled,
                random_seed=0,
                k=8,
                round_size=2,
                rounds=10,
                grid_convergence=True,
                grid_convergence_grid_bins=2,
                grid_convergence_min_rounds=2,
                grid_convergence_patience=1,
                grid_convergence_min_coverage_gain=1e-6,
                grid_convergence_min_novelty=0.25,
            )
            self.assertTrue(out["metrics"]["stopped_by_convergence"])
            self.assertTrue((out["round_dir"] / "round_001_indices.npy").exists())
