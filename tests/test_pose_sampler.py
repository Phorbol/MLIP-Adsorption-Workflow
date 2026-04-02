import unittest

import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc110, fcc111, molecule
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

    def test_pose_sampler_covers_multicenter_sites_for_monatomic_adsorbate(self):
        cases = [
            (fcc100("Pt", size=(4, 4, 4), vacuum=12.0), 3),
            (fcc110("Pt", size=(4, 4, 4), vacuum=12.0), 4),
        ]
        ads = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        cfg = PoseSamplerConfig(
            n_rotations=1,
            n_azimuth=1,
            n_shifts=1,
            shift_radius=0.0,
            min_height=0.8,
            max_height=2.2,
            height_step=0.1,
            max_poses_per_site=2,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
        )
        sampler = PoseSampler(cfg)
        for slab, expected_n_sites in cases:
            ctx = self.pre.build_context(slab)
            primitives = self.builder.build(slab, ctx)
            out = sampler.sample(slab=slab, adsorbate=ads, primitives=primitives, surface_atom_ids=ctx.detection.surface_atom_ids)
            self.assertEqual(len({int(p.primitive_index) for p in out}), expected_n_sites)

    def test_linear_adsorbate_site_oriented_quaternions_include_surface_normal(self):
        ads = molecule("CO")
        ads.positions = ads.positions - ads.get_center_of_mass()
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=2))
        quats = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="diatomic",
            rng=np.random.default_rng(0),
            n_rot=2,
        )
        self.assertEqual(len(quats), 2)
        vectors = []
        for q in quats:
            rotated = sampler._rotated_adsorbate(ads, q)
            axis = sampler._principal_axis(rotated)
            self.assertIsNotNone(axis)
            self.assertGreater(abs(float(np.dot(axis, np.array([0.0, 0.0, 1.0], dtype=float)))), 1.0 - 1e-6)
            vec = np.asarray(rotated.positions[1] - rotated.positions[0], dtype=float)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            vectors.append(vec)
        self.assertLess(float(np.dot(vectors[0], vectors[1])), -1.0 + 1e-6)

    def test_select_adsorption_origin_prefers_carbon_end_for_co(self):
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        self.assertEqual(PoseSampler._select_adsorption_origin_index(ads), 0)

    def test_pose_sampler_anchor_free_places_adsorbate_center_on_site_centerline(self):
        slab = fcc110("Pt", size=(4, 4, 4), vacuum=12.0)
        ctx = self.pre.build_context(slab)
        primitives = self.builder.build(slab, ctx)
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        cfg = PoseSamplerConfig(
            n_rotations=2,
            n_azimuth=2,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.2,
            max_height=3.2,
            height_step=0.1,
            placement_mode="anchor_free",
            max_poses_per_site=8,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
        )
        out = PoseSampler(cfg).sample(slab=slab, adsorbate=ads, primitives=[primitives[3]], surface_atom_ids=ctx.detection.surface_atom_ids)
        self.assertGreater(len(out), 0)
        primitive = primitives[3]
        found = False
        for cand in out:
            offset = np.asarray(cand.com, dtype=float) - np.asarray(primitive.center, dtype=float)
            lateral = offset - float(np.dot(offset, primitive.normal)) * np.asarray(primitive.normal, dtype=float)
            if np.linalg.norm(lateral) < 1e-6:
                found = True
                break
        self.assertTrue(found)

    def test_pose_sampler_anchoraware_places_selected_origin_atom_on_site_centerline(self):
        slab = fcc110("Pt", size=(4, 4, 4), vacuum=12.0)
        ctx = self.pre.build_context(slab)
        primitives = self.builder.build(slab, ctx)
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        cfg = PoseSamplerConfig(
            placement_mode="anchor_aware",
            n_rotations=2,
            n_azimuth=2,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.2,
            max_height=3.2,
            height_step=0.1,
            max_poses_per_site=8,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
        )
        out = PoseSampler(cfg).sample(slab=slab, adsorbate=ads, primitives=[primitives[3]], surface_atom_ids=ctx.detection.surface_atom_ids)
        self.assertGreater(len(out), 0)
        primitive = primitives[3]
        origin = PoseSampler._select_adsorption_origin_index(ads)
        found = False
        for cand in out:
            c_pos = np.asarray(cand.atoms.positions[int(origin)], dtype=float)
            offset = c_pos - np.asarray(primitive.center, dtype=float)
            lateral = offset - float(np.dot(offset, primitive.normal)) * np.asarray(primitive.normal, dtype=float)
            if np.linalg.norm(lateral) < 1e-6:
                found = True
                break
        self.assertTrue(found)

    def test_anchor_free_centering_uses_center_of_mass_by_default(self):
        ads = molecule("NH3")
        sampler = PoseSampler(PoseSamplerConfig(placement_mode="anchor_free", anchor_free_reference="center_of_mass"))
        centered = sampler._center_adsorbate_for_sampling(ads)
        mass_center = np.average(centered, axis=0, weights=ads.get_masses())
        self.assertTrue(np.allclose(mass_center, np.zeros(3), atol=1e-10))

    def test_anchoraware_centering_keeps_selected_origin_at_zero(self):
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        sampler = PoseSampler(PoseSamplerConfig(placement_mode="anchor_aware"))
        centered = sampler._center_adsorbate_for_sampling(ads)
        origin = PoseSampler._select_adsorption_origin_index(ads)
        self.assertTrue(np.allclose(centered[int(origin)], np.zeros(3), atol=1e-12))

    def test_linear_multicenter_sites_use_conservative_height_floor(self):
        slab = fcc110("Pt", size=(4, 4, 4), vacuum=12.0)
        ctx = self.pre.build_context(slab)
        primitives = self.builder.build(slab, ctx)
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        cfg = PoseSamplerConfig(
            n_rotations=2,
            n_azimuth=2,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.2,
            max_height=3.2,
            height_step=0.1,
            max_poses_per_site=8,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
        )
        out = PoseSampler(cfg).sample(slab=slab, adsorbate=ads, primitives=[primitives[3]], surface_atom_ids=ctx.detection.surface_atom_ids)
        self.assertGreater(len(out), 0)
        self.assertGreaterEqual(min(float(p.height) for p in out), 1.45 - 1e-8)

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
