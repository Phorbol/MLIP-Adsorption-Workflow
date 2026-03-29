import unittest

import numpy as np

from adsorption_ensemble.selection.strategies import FarthestPointSamplingSelector


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

