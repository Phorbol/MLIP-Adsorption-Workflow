import unittest
from unittest import mock

import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc110, fcc111, molecule
from ase.data import covalent_radii
from ase.geometry.geometry import general_find_mic

from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder
from adsorption_ensemble.site.embedding import PrimitiveEmbedder, PrimitiveEmbeddingConfig
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

    def _prepare_planar_basis_inputs(self):
        slab = fcc111("Pd", size=(3, 3, 4), vacuum=20.0)
        ctx = self.pre.build_context(slab)
        raw_primitives = self.builder.build(slab, ctx)
        embed = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.20)).fit_transform(
            slab=slab,
            primitives=list(raw_primitives),
            atom_features=np.asarray(slab.get_atomic_numbers(), dtype=float).reshape(-1, 1),
        )
        ads = molecule("C6H6")
        return slab, ads, ctx.detection.surface_atom_ids, list(embed.basis_primitives)

    @staticmethod
    def _ring_plane_angle_deg(adsorbate):
        pos = np.asarray(adsorbate.get_positions(), dtype=float)
        z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
        carbon_idx = [i for i, zi in enumerate(z) if int(zi) == 6]
        carbon_pos = pos[carbon_idx]
        center = np.mean(carbon_pos, axis=0, keepdims=True)
        centered = carbon_pos - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane_normal = np.asarray(vh[-1], dtype=float)
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
        c = abs(float(np.dot(plane_normal, np.array([0.0, 0.0, 1.0], dtype=float))))
        c = min(1.0, max(-1.0, c))
        return float(np.degrees(np.arccos(c)))

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

    def test_pose_sampler_adaptive_height_fallback_recovers_zero_pose_primitive(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        base_cfg = PoseSamplerConfig(
            n_rotations=1,
            n_azimuth=1,
            n_shifts=1,
            shift_radius=0.0,
            n_height_shifts=1,
            min_height=1.2,
            max_height=1.2,
            height_step=0.1,
            max_poses_per_site=None,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
            adaptive_height_fallback=False,
            adaptive_height_fallback_step=0.2,
            adaptive_height_fallback_max_extra=0.8,
            adaptive_height_fallback_contact_slack=0.4,
        )
        adaptive_cfg = PoseSamplerConfig(**{**vars(base_cfg), "adaptive_height_fallback": True})

        def fake_solve_height(self, **kwargs):
            return 1.2

        def fake_check_clash(self, placed_pos, tau):
            return bool(np.min(np.asarray(placed_pos, dtype=float)[:, 2]) < 20.0)

        with mock.patch.object(PoseSampler, "_solve_height", fake_solve_height), mock.patch.object(
            PoseSampler, "_check_clash_positions", fake_check_clash
        ):
            out_base = PoseSampler(base_cfg).sample(
                slab=slab,
                adsorbate=ads,
                primitives=[primitives[0]],
                surface_atom_ids=surface_ids,
            )
            out_adaptive = PoseSampler(adaptive_cfg).sample(
                slab=slab,
                adsorbate=ads,
                primitives=[primitives[0]],
                surface_atom_ids=surface_ids,
            )
        self.assertEqual(len(out_base), 0)
        self.assertGreaterEqual(len(out_adaptive), 1)
        self.assertGreater(out_adaptive[0].height, base_cfg.max_height)

    def test_linear_quaternion_schedule_spans_high_tilts_with_small_rotation_budget(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("C2H2")
        ctx = self.pre.build_context(slab)
        primitive = self.builder.build(slab, ctx)[0]
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=4, random_seed=0))
        mol_class = sampler._classify_molecule_shape(ads)
        quats = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads.copy(),
            normal=np.asarray(primitive.normal, dtype=float),
            mol_class=mol_class,
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        tilts = []
        for quat in quats:
            rotated = sampler._rotated_adsorbate(ads.copy(), quat)
            tilts.append(sampler._estimate_tilt_deg(rotated, np.asarray(primitive.normal, dtype=float)))
        self.assertEqual(len(quats), 4)
        self.assertGreaterEqual(max(tilts), 45.0)
        self.assertGreaterEqual(sum(1 for t in tilts if t >= 30.0), 2)

    def test_planar_like_detection_targets_benzene_but_not_nh3(self):
        sampler = PoseSampler(PoseSamplerConfig())
        self.assertTrue(bool(sampler._is_planar_like_nonlinear(molecule("C6H6"))))
        self.assertFalse(bool(sampler._is_planar_like_nonlinear(molecule("NH3"))))

    def test_planar_nonlinear_sampling_preserves_flat_tilt_upright_families_under_production_budget(self):
        slab, ads, surface_ids, primitives = self._prepare_planar_basis_inputs()
        cfg = PoseSamplerConfig(
            placement_mode="anchor_free",
            n_rotations=4,
            n_azimuth=8,
            n_shifts=2,
            shift_radius=0.15,
            min_height=1.2,
            max_height=3.4,
            height_step=0.10,
            max_poses_per_site=4,
            random_seed=0,
            adaptive_height_fallback=True,
            adaptive_height_fallback_step=0.20,
            adaptive_height_fallback_max_extra=1.60,
            adaptive_height_fallback_contact_slack=0.60,
        )
        out = PoseSampler(cfg).sample(
            slab=slab,
            adsorbate=ads,
            primitives=primitives,
            surface_atom_ids=surface_ids,
        )
        self.assertGreater(len(out), 0)
        angles = [self._ring_plane_angle_deg(p.atoms) for p in out]
        self.assertTrue(any(a < 20.0 for a in angles), msg=f"Missing flat family: {angles}")
        self.assertTrue(any(20.0 <= a < 70.0 for a in angles), msg=f"Missing tilted family: {angles}")
        self.assertTrue(any(a >= 70.0 for a in angles), msg=f"Missing upright family: {angles}")

    def test_pose_sampler_adaptive_height_fallback_recovers_clashing_candidate_even_when_some_exist(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        base_cfg = PoseSamplerConfig(
            n_rotations=2,
            n_azimuth=1,
            n_shifts=1,
            shift_radius=0.0,
            n_height_shifts=1,
            min_height=1.2,
            max_height=1.2,
            height_step=0.1,
            max_poses_per_site=None,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
            adaptive_height_fallback=False,
            adaptive_height_fallback_step=0.2,
            adaptive_height_fallback_max_extra=0.8,
            adaptive_height_fallback_contact_slack=0.4,
        )
        adaptive_cfg = PoseSamplerConfig(**{**vars(base_cfg), "adaptive_height_fallback": True})
        q_ident = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        def fake_quats(self, adsorbate_centered, normal, mol_class, rng, n_rot):
            return [q_ident.copy(), q_ident.copy()]

        def fake_solve_height(self, **kwargs):
            return 1.2

        calls = {"n": 0}

        def fake_check_clash(self, placed_pos, tau):
            calls["n"] += 1
            if calls["n"] == 2:
                return True
            return False

        def fake_min_dist(self, placed_pos, site_ids_local):
            return 1.2, 1.2

        with (
            mock.patch.object(PoseSampler, "_build_site_oriented_quaternions", fake_quats),
            mock.patch.object(PoseSampler, "_solve_height", fake_solve_height),
            mock.patch.object(PoseSampler, "_check_clash_positions", fake_check_clash),
            mock.patch.object(PoseSampler, "_min_scaled_distance_site_and_surface", fake_min_dist),
        ):
            out_base = PoseSampler(base_cfg).sample(
                slab=slab,
                adsorbate=ads,
                primitives=[primitives[0]],
                surface_atom_ids=surface_ids,
            )
            calls["n"] = 0
            out_adaptive = PoseSampler(adaptive_cfg).sample(
                slab=slab,
                adsorbate=ads,
                primitives=[primitives[0]],
                surface_atom_ids=surface_ids,
            )
        self.assertEqual(len(out_base), 1)
        self.assertEqual(len(out_adaptive), 2)

    def test_pose_sampler_covers_multicenter_site_kinds_for_monatomic_adsorbate(self):
        cases = [
            (fcc100("Pt", size=(4, 4, 4), vacuum=12.0), {"1c", "2c", "4c"}),
            (fcc110("Pt", size=(4, 4, 4), vacuum=12.0), {"1c", "2c", "4c"}),
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
            sampled_kinds = {primitives[int(p.primitive_index)].kind for p in out}
            self.assertSetEqual(sampled_kinds, expected_n_sites)

    def test_pose_sampler_orthogonal_fast_path_hook_preserves_results(self):
        slab = fcc100("Pt", size=(3, 3, 4), vacuum=12.0)
        ctx = self.pre.build_context(slab)
        primitives = self.builder.build(slab, ctx)
        ads = molecule("CO")
        cfg = PoseSamplerConfig(
            n_rotations=2,
            n_azimuth=4,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.0,
            max_height=2.8,
            height_step=0.1,
            random_seed=0,
            max_poses_per_site=3,
        )
        baseline = PoseSampler(cfg).sample(
            slab=slab,
            adsorbate=ads,
            primitives=primitives[:2],
            surface_atom_ids=ctx.detection.surface_atom_ids,
        )
        calls = {"n": 0}

        def fake_kernel(placed_pos, surf_frac, inv_cell, lengths, pbc_mask, denom2, site_ids_local):
            calls["n"] += 1
            ads_frac = np.asarray(placed_pos @ inv_cell, dtype=float)
            df = surf_frac[None, :, :] - ads_frac[:, None, :]
            for ax in range(3):
                if bool(pbc_mask[ax]):
                    df[:, :, ax] = df[:, :, ax] - np.round(df[:, :, ax])
            d0 = df[:, :, 0] * float(lengths[0])
            d1 = df[:, :, 1] * float(lengths[1])
            d2 = df[:, :, 2] * float(lengths[2])
            dist2 = d0 * d0 + d1 * d1 + d2 * d2
            min_surface2 = float(np.min(dist2 / denom2))
            if len(site_ids_local) == 0:
                return np.inf, min_surface2
            cols = np.asarray(site_ids_local, dtype=int)
            min_site2 = float(np.min(dist2[:, cols] / denom2[:, cols]))
            return min_site2, min_surface2

        with mock.patch("adsorption_ensemble.pose.sampler._get_pose_orthogonal_min_ratio_kernel", return_value=fake_kernel):
            accelerated = PoseSampler(cfg).sample(
                slab=slab,
                adsorbate=ads,
                primitives=primitives[:2],
                surface_atom_ids=ctx.detection.surface_atom_ids,
            )
        self.assertGreater(calls["n"], 0)
        self.assertEqual(len(accelerated), len(baseline))
        self.assertEqual([int(p.primitive_index) for p in accelerated], [int(p.primitive_index) for p in baseline])
        self.assertTrue(np.allclose([float(p.height) for p in accelerated], [float(p.height) for p in baseline], atol=1e-8))

    def test_linear_adsorbate_site_oriented_quaternions_include_surface_normal_with_small_budget(self):
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
        alignments = []
        for q in quats:
            rotated = sampler._rotated_adsorbate(ads, q)
            axis = sampler._principal_axis(rotated)
            self.assertIsNotNone(axis)
            alignments.append(abs(float(np.dot(axis, np.array([0.0, 0.0, 1.0], dtype=float)))))
            vec = np.asarray(rotated.positions[1] - rotated.positions[0], dtype=float)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            vectors.append(vec)
        self.assertTrue(any(v >= 0.99 for v in alignments))
        self.assertTrue(any(0.80 <= v <= 0.99 for v in alignments))
        self.assertLess(float(np.dot(vectors[0], vectors[1])), -0.80)

    def test_linear_adsorbate_site_oriented_quaternions_include_tilted_states_when_budget_allows(self):
        ads = molecule("CO")
        ads.positions = ads.positions - ads.get_center_of_mass()
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=4))
        quats = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="diatomic",
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        tilts = []
        for q in quats:
            rotated = sampler._rotated_adsorbate(ads, q)
            tilts.append(float(sampler._estimate_tilt_deg(rotated, np.array([0.0, 0.0, 1.0], dtype=float))))
        self.assertTrue(any(t <= 1e-3 for t in tilts))
        self.assertTrue(any(t >= 10.0 for t in tilts))

    def test_nonlinear_adsorbate_body_frame_quaternions_are_deterministic_across_rng(self):
        ads = molecule("H2O")
        ads.positions = ads.positions - ads.get_center_of_mass()
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=4, nonlinear_atom_down_coverage=False))
        q1 = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="nonlinear",
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        q2 = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="nonlinear",
            rng=np.random.default_rng(123),
            n_rot=4,
        )
        self.assertEqual(len(q1), len(q2))
        self.assertTrue(all(float(sampler._quaternion_distance(a, b)) < 1e-8 for a, b in zip(q1, q2, strict=False)))

    def test_nonlinear_adsorbate_body_frame_schedule_is_input_rotation_invariant(self):
        ads = molecule("H2O")
        ads.positions = ads.positions - ads.get_center_of_mass()
        rot = PoseSampler._axis_angle_to_matrix(np.array([0.3, -0.5, 0.8], dtype=float), 0.9)
        ads_rot = ads.copy()
        ads_rot.positions = np.asarray(ads.positions, dtype=float) @ rot.T
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=4, nonlinear_atom_down_coverage=False))
        q1 = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="nonlinear",
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        q2 = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads_rot,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="nonlinear",
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        self.assertEqual(len(q1), len(q2))
        for qa, qb in zip(q1, q2, strict=False):
            pa = sampler._rotated_adsorbate(ads, qa).positions
            pb = sampler._rotated_adsorbate(ads_rot, qb).positions
            self.assertTrue(np.allclose(pa, pb, atol=1e-8))

    def test_nonlinear_quaternion_schedule_includes_hetero_atom_down_for_methanol(self):
        ads = molecule("CH3OH")
        ads.positions = ads.positions - ads.get_center_of_mass()
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=4))
        quats = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="nonlinear",
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        lowest_symbols = []
        for q in quats:
            rotated = sampler._rotated_adsorbate(ads, q)
            lowest_idx = int(np.argmin(np.asarray(rotated.positions, dtype=float)[:, 2]))
            lowest_symbols.append(str(ads[lowest_idx].symbol))
        self.assertIn("O", lowest_symbols)

    def test_nonlinear_quaternion_schedule_keeps_backbone_coverage_when_atom_down_budget_is_full(self):
        ads = Atoms(
            "C5",
            positions=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.2],
                    [0.0, -1.0, -0.3],
                    [0.2, 0.1, 1.1],
                ],
                dtype=float,
            ),
        )
        ads.positions = ads.positions - ads.get_center_of_mass()
        sampler = PoseSampler(PoseSamplerConfig(n_rotations=4, nonlinear_atom_down_coverage=True))
        atom_down = sampler._nonlinear_atom_down_quaternions(
            adsorbate_centered=ads,
            max_vectors=4,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        self.assertEqual(len(atom_down), 4)
        quats = sampler._build_site_oriented_quaternions(
            adsorbate_centered=ads,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mol_class="nonlinear",
            rng=np.random.default_rng(0),
            n_rot=4,
        )
        self.assertEqual(len(quats), 4)
        self.assertTrue(
            any(
                all(float(sampler._quaternion_distance(quat, ref)) > 1e-8 for ref in atom_down)
                for quat in quats
            ),
            msg="Nonlinear coverage collapsed to atom-down quaternions only; no backbone orientation survived.",
        )

    def test_tangent_shift_schedule_is_rng_invariant(self):
        s1 = PoseSampler._sample_tangent_shifts(np.random.default_rng(0), 4, 0.35)
        s2 = PoseSampler._sample_tangent_shifts(np.random.default_rng(123), 4, 0.35)
        self.assertEqual(len(s1), len(s2))
        self.assertTrue(all(np.allclose(a, b, atol=1e-12) for a, b in zip(s1, s2, strict=False)))

    def test_select_adsorption_origin_prefers_carbon_end_for_co(self):
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        self.assertEqual(PoseSampler._select_adsorption_origin_index(ads), 0)

    def test_select_adsorption_origin_prefers_heavy_donor_for_protic_adsorbates(self):
        cases = {
            "H2O": "O",
            "NH3": "N",
            "CH3OH": "O",
        }
        for name, expected_symbol in cases.items():
            ads = molecule(name)
            idx = PoseSampler._select_adsorption_origin_index(ads)
            self.assertEqual(ads[int(idx)].symbol, expected_symbol)

    def test_orient_adsorbate_for_binding_places_donor_at_origin_and_ligands_upward(self):
        for name in ("CO", "H2O", "NH3", "CH3OH"):
            ads = molecule(name)
            if name == "CO" and ads[0].symbol != "C":
                ads = ads[[1, 0]]
            idx = PoseSampler._select_adsorption_origin_index(ads)
            oriented = PoseSampler.orient_adsorbate_for_binding(ads, binding_atom_index=idx, normal=np.array([0.0, 0.0, 1.0], dtype=float))
            self.assertTrue(np.allclose(oriented.positions[int(idx)], np.zeros(3), atol=1e-10))
            other_ids = [i for i in range(len(oriented)) if i != int(idx)]
            if other_ids:
                self.assertGreater(float(np.mean(oriented.positions[other_ids, 2])), 0.0)

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

    def test_default_linear_sampling_keeps_tilted_candidates(self):
        slab, ads, surface_ids, primitives = self._prepare_inputs()
        cfg = PoseSamplerConfig(
            n_rotations=4,
            n_azimuth=8,
            n_shifts=2,
            shift_radius=0.15,
            min_height=1.2,
            max_height=3.4,
            height_step=0.1,
            max_poses_per_site=4,
            prune_com_distance=0.0,
            prune_rot_distance=0.0,
            random_seed=0,
        )
        out = PoseSampler(cfg).sample(slab=slab, adsorbate=ads, primitives=[primitives[0]], surface_atom_ids=surface_ids)
        self.assertGreater(len(out), 0)
        self.assertTrue(any(float(p.tilt_deg) >= 10.0 for p in out))

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
