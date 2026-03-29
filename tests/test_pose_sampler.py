import unittest

import numpy as np
from ase.build import fcc111, molecule
from ase.data import covalent_radii
from ase.geometry.geometry import general_find_mic

from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder
from adsorption_ensemble.surface import SurfacePreprocessor


class TestPoseSampler(unittest.TestCase):
    def setUp(self):
        self.pre = SurfacePreprocessor(min_surface_atoms=6)
        self.builder = PrimitiveBuilder()

    def _prepare_inputs(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ctx = self.pre.build_context(slab)
        primitives = self.builder.build(slab, ctx)
        ads = molecule("CO")
        return slab, ads, ctx.detection.surface_atom_ids, primitives

    @staticmethod
    def _min_scaled_distance(slab, adsorbate, surface_atom_ids):
        combined = slab + adsorbate
        n_slab = len(slab)
        ads_z = adsorbate.get_atomic_numbers()
        surf_z = slab.get_atomic_numbers()[surface_atom_ids]
        min_ratio = np.inf
        for i, z_ads in enumerate(ads_z):
            ri = covalent_radii[int(z_ads)]
            ci = n_slab + i
            for j_local, sid in enumerate(surface_atom_ids):
                rj = covalent_radii[int(surf_z[j_local])]
                d = combined.get_distance(ci, int(sid), mic=True)
                ratio = d / max(1e-12, (ri + rj))
                min_ratio = min(min_ratio, ratio)
        return float(min_ratio)

    def test_pose_sampler_generates_valid_poses(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        cfg = PoseSamplerConfig(
            n_rotations=4,
            n_shifts=3,
            shift_radius=0.2,
            min_height=1.0,
            max_height=3.2,
            height_step=0.08,
            random_seed=7,
            max_poses_per_site=3,
        )
        sampler = PoseSampler(cfg)
        poses = sampler.sample(slab=slab, adsorbate=ads, primitives=primitives[:2], surface_atom_ids=surface_ids)
        self.assertGreater(len(poses), 0)
        for p in poses:
            self.assertGreaterEqual(p.height, cfg.min_height - 1e-8)
            self.assertLessEqual(p.height, cfg.max_height + 1e-8)
            self.assertEqual(len(p.atoms), len(ads))
            min_ratio = self._min_scaled_distance(slab, p.atoms, surface_ids)
            self.assertGreaterEqual(min_ratio, cfg.clash_tau - 1e-3)

    def test_pose_sampler_reproducible_with_seed(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        cfg = PoseSamplerConfig(
            n_rotations=3,
            n_shifts=2,
            shift_radius=0.15,
            min_height=1.0,
            max_height=2.8,
            height_step=0.1,
            random_seed=42,
            max_poses_per_site=2,
        )
        s1 = PoseSampler(cfg)
        s2 = PoseSampler(cfg)
        out1 = s1.sample(slab=slab, adsorbate=ads, primitives=primitives[:3], surface_atom_ids=surface_ids)
        out2 = s2.sample(slab=slab, adsorbate=ads, primitives=primitives[:3], surface_atom_ids=surface_ids)
        self.assertEqual(len(out1), len(out2))
        self.assertGreater(len(out1), 0)
        self.assertTrue(np.allclose(out1[0].quaternion, out2[0].quaternion))
        self.assertTrue(np.allclose(out1[0].com, out2[0].com))
        self.assertAlmostEqual(out1[0].height, out2[0].height, places=8)

    def test_pose_sampler_prunes_duplicate_initial_poses(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        dup = [primitives[0], primitives[0]]
        cfg = PoseSamplerConfig(
            n_rotations=1,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.0,
            max_height=2.6,
            height_step=0.1,
            prune_com_distance=10.0,
            prune_rot_distance=10.0,
            random_seed=1,
            max_poses_per_site=None,
        )
        sampler = PoseSampler(cfg)
        out = sampler.sample(slab=slab, adsorbate=ads, primitives=dup, surface_atom_ids=surface_ids)
        self.assertEqual(len(out), 1)

    def test_pose_sampler_has_azimuth_diversity_near_single_site(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        cfg = PoseSamplerConfig(
            n_rotations=1,
            n_azimuth=12,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.0,
            max_height=3.0,
            height_step=0.08,
            random_seed=0,
            max_poses_per_site=12,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
        )
        sampler = PoseSampler(cfg)
        out = sampler.sample(slab=slab, adsorbate=ads, primitives=[primitives[0]], surface_atom_ids=surface_ids)
        self.assertGreater(len(out), 0)
        azimuth_ids = {int(p.azimuth_index) for p in out}
        self.assertGreaterEqual(len(azimuth_ids), 6)

    def test_pose_sampler_reduces_rotation_budget_for_monatomic(self):
        slab, _, surface_ids, primitives = self._prepare_inputs()
        ads = molecule("Al")
        cfg = PoseSamplerConfig(
            n_rotations=8,
            n_azimuth=12,
            n_shifts=1,
            max_poses_per_site=6,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
        )
        sampler = PoseSampler(cfg)
        out = sampler.sample(slab=slab, adsorbate=ads, primitives=[primitives[0]], surface_atom_ids=surface_ids)
        self.assertGreater(len(out), 0)
        self.assertEqual({int(p.azimuth_index) for p in out}, {0})
        self.assertEqual({int(p.rotation_index) for p in out}, {0})

    def test_pose_sampler_supports_height_shift_sampling(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        cfg = PoseSamplerConfig(
            n_rotations=1,
            n_azimuth=1,
            n_shifts=1,
            n_height_shifts=3,
            height_shift_step=0.12,
            min_height=1.0,
            max_height=3.0,
            max_poses_per_site=12,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=3,
        )
        out = PoseSampler(cfg).sample(slab=slab, adsorbate=ads, primitives=[primitives[0]], surface_atom_ids=surface_ids)
        self.assertGreater(len(out), 0)
        self.assertGreaterEqual(len({int(p.height_shift_index) for p in out}), 2)
        self.assertGreaterEqual(len({round(float(p.height), 3) for p in out}), 2)

    def test_triclinic_mic_cache_matches_ase_general_find_mic(self):
        cell = np.array([[2.8, 0.2, 0.0], [0.9, 2.6, 0.0], [0.1, 0.3, 14.0]], dtype=float)
        pbc = np.array([True, True, False], dtype=bool)
        v = np.random.default_rng(0).normal(size=(400, 3)).astype(float) * 6.0
        sampler = PoseSampler(PoseSamplerConfig())
        sampler._cell = cell
        sampler._pbc = pbc
        sampler._cell_orthogonal = False
        sampler._prepare_mic_cache()
        self.assertIsNotNone(sampler._mic_rcell)
        d2_cache = sampler._mic_minlen2_cached(v)
        _, d_ref = general_find_mic(v, cell, pbc=pbc)
        d2_ref = np.asarray(d_ref, dtype=float) ** 2
        self.assertLessEqual(float(np.max(np.abs(d2_cache - d2_ref))), 1e-10)


if __name__ == "__main__":
    unittest.main()
