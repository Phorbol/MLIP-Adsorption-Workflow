import unittest

import numpy as np
from ase import Atoms

from adsorption_ensemble.selection import StageSelectionConfig, apply_stage_selection


class TestStageSelection(unittest.TestCase):
    def _frames(self):
        base = []
        for stretch in [0.0, 0.02, 0.30, 0.32]:
            a = Atoms(
                "PtPtHHO",
                positions=[
                    [0, 0, 0],
                    [2, 0, 0],
                    [0.0, 0.0, 1.5],
                    [0.95 + stretch, 0.0, 1.5],
                    [0.3, 0.8 + stretch, 1.5],
                ],
            )
            base.append(a)
        return base

    def test_fps_selection_reduces_pool(self):
        frames = self._frames()
        ids, diag = apply_stage_selection(
            frames=frames,
            config=StageSelectionConfig(enabled=True, strategy="fps", max_candidates=2, random_seed=0, descriptor="adsorbate_pair_distance"),
            slab_n=2,
        )
        self.assertEqual(len(ids), 2)
        self.assertEqual(diag["n_selected"], 2)

    def test_energy_rmsd_window_prefers_low_energy_representatives(self):
        frames = self._frames()
        energies = np.asarray([0.0, 0.02, 0.30, 0.32], dtype=float)
        ids, diag = apply_stage_selection(
            frames=frames,
            config=StageSelectionConfig(
                enabled=True,
                strategy="energy_rmsd_window",
                energy_window_ev=0.05,
                rmsd_threshold=0.05,
                descriptor="adsorbate_pair_distance",
            ),
            slab_n=2,
            energies=energies,
        )
        self.assertEqual(ids, [0])
        self.assertEqual(diag["n_energy_window_keep"], 2)

    def test_cluster_selection_requires_no_k_and_returns_representatives(self):
        frames = self._frames()
        energies = np.asarray([0.1, 0.0, 0.4, 0.3], dtype=float)
        ids, diag = apply_stage_selection(
            frames=frames,
            config=StageSelectionConfig(
                enabled=True,
                strategy="hierarchical",
                cluster_threshold=0.05,
                descriptor="adsorbate_pair_distance",
            ),
            slab_n=2,
            energies=energies,
        )
        self.assertEqual(ids, [1, 3])
        self.assertEqual(diag["n_clusters"], 2)

    def test_surface_distance_descriptor_distinguishes_rigid_adsorption_sites(self):
        frames = []
        slab = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [1.25, 2.0, 0.0],
            ],
            dtype=float,
        )
        ads_a = np.asarray([[0.0, 0.0, 1.8], [0.0, 0.0, 2.95]], dtype=float)
        ads_b = np.asarray([[1.25, 0.8, 1.8], [1.25, 0.8, 2.95]], dtype=float)
        for ads in (ads_a, ads_b):
            frames.append(Atoms("PtPtPtCO", positions=np.vstack([slab, ads])))
        pair_ids, _ = apply_stage_selection(
            frames=frames,
            config=StageSelectionConfig(
                enabled=True,
                strategy="hierarchical",
                cluster_threshold=0.05,
                descriptor="adsorbate_pair_distance",
            ),
            slab_n=3,
        )
        surface_ids, diag = apply_stage_selection(
            frames=frames,
            config=StageSelectionConfig(
                enabled=True,
                strategy="hierarchical",
                cluster_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            slab_n=3,
        )
        self.assertEqual(pair_ids, [0])
        self.assertEqual(surface_ids, [0, 1])
        self.assertEqual(diag["n_clusters"], 2)


if __name__ == "__main__":
    unittest.main()
