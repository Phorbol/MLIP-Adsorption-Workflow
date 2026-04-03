import unittest
from pathlib import Path

from adsorption_ensemble.workflows import (
    list_sampling_schedule_presets,
    make_adsorption_workflow_config,
    make_pose_sampler_config,
    make_sampling_schedule,
)


class TestWorkflowPresets(unittest.TestCase):
    def test_anchorfree_pose_preset_is_default(self):
        cfg = make_pose_sampler_config()
        self.assertEqual(cfg.placement_mode, "anchor_free")
        self.assertEqual(cfg.anchor_free_reference, "center_of_mass")

    def test_anchoraware_preset_can_be_requested(self):
        cfg = make_pose_sampler_config(placement_mode="anchor_aware", exhaustive=True)
        self.assertEqual(cfg.placement_mode, "anchor_aware")
        self.assertGreaterEqual(int(cfg.max_poses_per_site or 0), 12)

    def test_workflow_preset_propagates_pose_and_basin_modes(self):
        cfg = make_adsorption_workflow_config(
            Path("artifacts") / "preset_smoke",
            placement_mode="anchor_free",
            dedup_metric="rmsd",
            signature_mode="provenance",
        )
        self.assertEqual(cfg.pose_sampler_config.placement_mode, "anchor_free")
        self.assertEqual(cfg.basin_config.signature_mode, "provenance")
        self.assertEqual(cfg.basin_config.dedup_metric, "rmsd")
        self.assertEqual(cfg.basin_config.desorption_min_bonds, 1)

    def test_sampling_schedule_default_is_conservative_multistage(self):
        sched = make_sampling_schedule()
        self.assertEqual(sched.name, "multistage_default")
        self.assertTrue(sched.pre_relax_selection.enabled)
        self.assertEqual(sched.pre_relax_selection.strategy, "fps")
        self.assertEqual(sched.pre_relax_selection.max_candidates, 24)
        self.assertEqual(sched.pre_relax_selection.descriptor, "adsorbate_surface_distance")
        self.assertTrue(sched.post_relax_selection.enabled)
        self.assertEqual(sched.post_relax_selection.strategy, "energy_rmsd_window")
        self.assertAlmostEqual(float(sched.post_relax_selection.energy_window_ev), 3.0)
        self.assertAlmostEqual(float(sched.post_relax_selection.rmsd_threshold), 0.05)
        self.assertEqual(sched.post_relax_selection.descriptor, "adsorbate_surface_distance")

    def test_sampling_schedule_aliases_and_listing(self):
        presets = list_sampling_schedule_presets()
        self.assertIn("multistage_default", presets)
        self.assertIn("multistage_iterative_fps_grid", presets)
        self.assertIn("no_selection", presets)
        sched = make_sampling_schedule("molclus_like_default")
        self.assertEqual(sched.name, "multistage_default")
        adaptive = make_sampling_schedule("adaptive_default")
        self.assertEqual(adaptive.name, "multistage_iterative_fps_grid")
        self.assertEqual(adaptive.pre_relax_selection.strategy, "iterative_fps")
        self.assertTrue(adaptive.pre_relax_selection.grid_convergence)
        off = make_sampling_schedule("none")
        self.assertFalse(off.pre_relax_selection.enabled)
        self.assertFalse(off.post_relax_selection.enabled)


if __name__ == "__main__":
    unittest.main()
