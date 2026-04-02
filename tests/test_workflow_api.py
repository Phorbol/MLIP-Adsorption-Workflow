import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ase.build import fcc111, molecule

from adsorption_ensemble.selection import StageSelectionConfig
from adsorption_ensemble.workflows import SamplingSchedule, generate_adsorption_ensemble


class TestWorkflowAPI(unittest.TestCase):
    def test_generate_adsorption_ensemble_anchorfree(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("CO")
        with TemporaryDirectory() as td:
            result = generate_adsorption_ensemble(
                slab=slab,
                adsorbate=ads,
                work_dir=Path(td) / "anchorfree",
                placement_mode="anchor_free",
                schedule=SamplingSchedule(name="smoke", exhaustive_pose_sampling=False),
            )
            self.assertEqual(result.summary["placement_mode"], "anchor_free")
            self.assertGreater(result.summary["n_basis_primitives"], 0)
            self.assertGreater(result.summary["n_pose_frames"], 0)
            self.assertTrue(Path(result.files["basins_json"]).exists())
            self.assertTrue(Path(result.files["nodes_json"]).exists())

    def test_generate_adsorption_ensemble_anchoraware(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("CO")
        with TemporaryDirectory() as td:
            result = generate_adsorption_ensemble(
                slab=slab,
                adsorbate=ads,
                work_dir=Path(td) / "anchoraware",
                placement_mode="anchor_aware",
                schedule=SamplingSchedule(name="smoke", exhaustive_pose_sampling=False),
            )
            self.assertEqual(result.summary["placement_mode"], "anchor_aware")
            self.assertGreaterEqual(result.summary["n_basins"], 1)

    def test_generate_adsorption_ensemble_accepts_distinct_stage_selectors(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("CO")
        with TemporaryDirectory() as td:
            result = generate_adsorption_ensemble(
                slab=slab,
                adsorbate=ads,
                work_dir=Path(td) / "scheduled",
                placement_mode="anchor_free",
                schedule=SamplingSchedule(
                    name="scheduled",
                    exhaustive_pose_sampling=False,
                    pre_relax_selection=StageSelectionConfig(enabled=True, strategy="fps", max_candidates=4, random_seed=0),
                    post_relax_selection=StageSelectionConfig(enabled=True, strategy="hierarchical", cluster_threshold=0.05),
                ),
            )
            self.assertLessEqual(result.summary["n_pose_frames_selected_for_basin"], result.summary["n_pose_frames"])
            self.assertEqual(result.workflow.summary["pre_relax_selection"]["strategy"], "fps")
            self.assertEqual(result.workflow.basin_result.summary["post_relax_selection"]["strategy"], "hierarchical")


if __name__ == "__main__":
    unittest.main()
