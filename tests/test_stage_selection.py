import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_iterative_fps_grid_convergence_writes_round_artifacts(self):
        frames = []
        for idx in range(12):
            atoms = Atoms(
                "PtPtHHO",
                positions=[
                    [0, 0, 0],
                    [2, 0, 0],
                    [0.0 + 0.02 * (idx % 2), 0.0, 1.5],
                    [0.95 + 0.02 * (idx % 2), 0.0, 1.5],
                    [0.3, 0.8 + 0.02 * (idx // 2 % 2), 1.5],
                ],
                cell=[4.0, 4.0, 8.0],
                pbc=[True, True, False],
            )
            atoms.info["basis_id"] = idx % 2
            atoms.info["primitive_index"] = idx
            frames.append(atoms)
        with TemporaryDirectory() as td:
            ids, diag = apply_stage_selection(
                frames=frames,
                config=StageSelectionConfig(
                    enabled=True,
                    strategy="iterative_fps",
                    max_candidates=10,
                    descriptor="adsorbate_surface_distance",
                    random_seed=0,
                    fps_round_size=2,
                    fps_rounds=8,
                    grid_convergence=True,
                    grid_convergence_grid_bins=2,
                    grid_convergence_min_rounds=2,
                    grid_convergence_patience=1,
                    grid_convergence_min_coverage_gain=1e-6,
                    grid_convergence_min_novelty=0.25,
                ),
                slab_n=2,
                artifacts_dir=Path(td) / "rounds",
            )
            self.assertGreaterEqual(len(ids), 2)
            self.assertTrue(diag["stopped_by_convergence"])
            self.assertTrue((Path(diag["round_dir"]) / "round_001_indices.npy").exists())
            self.assertIn("convergence", diag["metrics"])

    def test_iterative_fps_site_occupancy_convergence_uses_metadata_bins(self):
        frames = []
        for idx in range(12):
            atoms = Atoms(
                "PtPtHHO",
                positions=[
                    [0, 0, 0],
                    [2, 0, 0],
                    [0.0 + 0.02 * (idx % 3), 0.0, 1.5],
                    [0.95 + 0.02 * (idx % 3), 0.0, 1.5],
                    [0.3, 0.8, 1.5],
                ],
            )
            atoms.info["basis_id"] = idx % 2
            atoms.info["primitive_index"] = idx % 2
            frames.append(atoms)
        ids, diag = apply_stage_selection(
            frames=frames,
            config=StageSelectionConfig(
                enabled=True,
                strategy="iterative_fps",
                max_candidates=10,
                descriptor="adsorbate_pair_distance",
                random_seed=0,
                fps_round_size=2,
                fps_rounds=8,
                occupancy_convergence=True,
                occupancy_bucket_keys=("basis_id",),
                occupancy_min_new_bins=0,
                occupancy_min_rounds=2,
                occupancy_patience=1,
            ),
            slab_n=2,
        )
        self.assertGreaterEqual(len(ids), 2)
        self.assertTrue(diag["stopped_by_convergence"])
        conv = diag["metrics"]["convergence"]
        self.assertEqual(conv["bucket_keys"], ["basis_id"])
        self.assertIn("round_metrics", conv)

    def test_surface_distance_descriptor_respects_pbc_equivalence(self):
        frame_a = Atoms(
            "PtPtH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.1, 0.0, 1.4],
            ],
            cell=[2.0, 2.0, 8.0],
            pbc=[True, True, False],
        )
        frame_b = frame_a.copy()
        frame_b.positions[-1, 0] += 2.0
        ids, diag = apply_stage_selection(
            frames=[frame_a, frame_b],
            config=StageSelectionConfig(
                enabled=True,
                strategy="hierarchical",
                cluster_threshold=1e-6,
                descriptor="adsorbate_surface_distance",
            ),
            slab_n=2,
        )
        self.assertEqual(ids, [0])
        self.assertEqual(diag["n_clusters"], 1)


if __name__ == "__main__":
    unittest.main()
