import unittest
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
from ase.build import fcc111, molecule

from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder
from adsorption_ensemble.surface import SurfacePreprocessor


class TestBasinPipeline(unittest.TestCase):
    def test_basin_builder_dedup_duplicates(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        pre = SurfacePreprocessor(min_surface_atoms=6)
        ctx = pre.build_context(slab)
        primitives = PrimitiveBuilder().build(slab, ctx)
        ads = molecule("H2O")
        sampler = PoseSampler(
            PoseSamplerConfig(
                n_rotations=2,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.8,
                max_height=2.6,
                height_step=0.2,
                random_seed=0,
                max_poses_per_site=2,
            )
        )
        poses = sampler.sample(slab=slab, adsorbate=ads, primitives=primitives[:2], surface_atom_ids=ctx.detection.surface_atom_ids)
        self.assertGreaterEqual(len(poses), 1)
        frame1 = slab + poses[0].atoms
        frame2 = frame1.copy()
        frame3 = frame1.copy()
        frame3.positions[len(slab) :] += np.array([1.6, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                relax_maxf=0.1,
                relax_steps=2,
                energy_window_ev=1.0,
                rmsd_threshold=1e-6,
                desorption_min_bonds=0,
                final_basin_merge_metric="off",
                work_dir=Path(td),
            )
            out = BasinBuilder(config=cfg).build(
                frames=[frame1, frame2, frame3],
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=int(ctx.classification.normal_axis),
            )
        self.assertEqual(out.summary["n_input"], 3)
        self.assertGreaterEqual(out.summary["n_basins"], 1)
        self.assertEqual(out.summary["n_basins"], 2)
        member_counts = sorted(len(b.member_candidate_ids) for b in out.basins)
        self.assertEqual(member_counts, [1, 2])

    def test_basin_builder_rejects_desorbed_candidates_when_required(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("CO")
        adsorbed = slab.copy() + ads.copy()
        adsorbed.positions[len(slab) :] += slab.positions[0] + np.array([0.0, 0.0, 1.85])
        desorbed = slab.copy() + ads.copy()
        desorbed.positions[len(slab) :] += np.array([0.0, 0.0, 8.0])
        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=2.0,
                rmsd_threshold=0.10,
                desorption_min_bonds=1,
                work_dir=Path(td),
            )
            out = BasinBuilder(config=cfg).build(
                frames=[adsorbed, desorbed],
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=2,
            )
        self.assertEqual(out.summary["n_input"], 2)
        self.assertEqual(out.summary["n_rejected"], 1)
        self.assertEqual(len(out.rejected), 1)
        self.assertEqual(out.rejected[0].reason, "desorption")
        self.assertEqual(out.summary["n_basins"], 1)

    def test_basin_builder_auto_ref_canonical_merge_skips_single_atom_adsorbates(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("H")
        frame_a = slab.copy() + ads.copy()
        frame_b = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] = slab.positions[0] + np.array([0.0, 0.0, 1.0])
        frame_b.positions[len(slab) :] = slab.positions[1] + np.array([0.0, 0.0, 1.0])
        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=2.0,
                dedup_metric="signature_only",
                signature_mode="absolute",
                desorption_min_bonds=1,
                final_basin_merge_metric="auto_ref_canonical_mace",
                final_basin_merge_node_l2_threshold=None,
                work_dir=Path(td),
            )
            out = BasinBuilder(config=cfg).build(
                frames=[frame_a, frame_b],
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=2,
            )
        self.assertEqual(out.summary["n_basins"], 2)
        self.assertEqual(out.summary["final_basin_merge"]["status"], "skipped_single_atom_adsorbate")
        self.assertEqual(out.summary["final_basin_merge"]["signature_mode"], "reference_canonical")

    def test_basin_builder_threads_final_merge_energy_gate(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("CO")
        frame_a = slab.copy() + ads.copy()
        frame_b = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] = slab.positions[0] + np.array([0.0, 0.0, 1.85])
        frame_b.positions[len(slab) :] = slab.positions[1] + np.array([0.0, 0.0, 1.85])

        def _fake_merge(*, basins, **kwargs):
            return basins, {
                "metric": "pure_mace",
                "n_input_basins": len(basins),
                "n_output_basins": len(basins),
                "energy_gate_ev": kwargs.get("energy_gate_ev"),
            }

        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=2.0,
                dedup_metric="signature_only",
                signature_mode="absolute",
                desorption_min_bonds=1,
                final_basin_merge_metric="mace_node_l2",
                final_basin_merge_node_l2_threshold=0.02,
                final_basin_merge_energy_gate_ev=0.05,
                work_dir=Path(td),
            )
            with mock.patch(
                "adsorption_ensemble.basin.pipeline.merge_basin_representatives_by_mace_node_l2",
                side_effect=_fake_merge,
            ) as patched_merge:
                out = BasinBuilder(config=cfg).build(
                    frames=[frame_a, frame_b],
                    slab_ref=slab,
                    adsorbate_ref=ads,
                    slab_n=len(slab),
                    normal_axis=2,
                )
        self.assertEqual(out.summary["n_basins"], 2)
        self.assertEqual(out.summary["final_basin_merge"]["status"], "ok")
        self.assertAlmostEqual(out.summary["final_basin_merge"]["energy_gate_ev"], 0.05)
        self.assertEqual(patched_merge.call_count, 1)
        self.assertAlmostEqual(patched_merge.call_args.kwargs["energy_gate_ev"], 0.05)

    def test_basin_builder_preserves_requested_final_merge_metric_in_summary(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("CO")
        frame_a = slab.copy() + ads.copy()
        frame_b = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] = slab.positions[0] + np.array([0.0, 0.0, 1.85])
        frame_b.positions[len(slab) :] = slab.positions[1] + np.array([0.0, 0.0, 1.85])

        def _fake_merge(*, basins, **kwargs):
            return basins, {
                "metric": "mace_node_l2",
                "n_input_basins": len(basins),
                "n_output_basins": len(basins),
            }

        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=2.0,
                dedup_metric="signature_only",
                signature_mode="absolute",
                desorption_min_bonds=1,
                mace_model_path="/tmp/fake.model",
                final_basin_merge_metric="auto_ref_canonical_mace",
                final_basin_merge_node_l2_threshold=0.02,
                work_dir=Path(td),
            )
            with mock.patch(
                "adsorption_ensemble.basin.pipeline.merge_basin_representatives_by_mace_node_l2",
                side_effect=_fake_merge,
            ):
                out = BasinBuilder(config=cfg).build(
                    frames=[frame_a, frame_b],
                    slab_ref=slab,
                    adsorbate_ref=ads,
                    slab_n=len(slab),
                    normal_axis=2,
                )
        self.assertEqual(out.summary["final_basin_merge"]["status"], "ok")
        self.assertEqual(out.summary["final_basin_merge"]["metric"], "auto_ref_canonical_mace")
        self.assertEqual(out.summary["final_basin_merge"]["backend_metric"], "mace_node_l2")
