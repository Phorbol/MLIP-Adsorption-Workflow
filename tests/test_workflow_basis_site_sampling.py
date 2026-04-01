import tempfile
import unittest
from pathlib import Path

from ase.build import fcc111, molecule

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.workflows.adsorption import AdsorptionWorkflowConfig, run_adsorption_workflow


class TestWorkflowBasisSiteSampling(unittest.TestCase):
    def test_workflow_samples_basis_not_raw_primitives(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=10.0)
        ads = molecule("CO")
        with tempfile.TemporaryDirectory() as td:
            cfg = AdsorptionWorkflowConfig(
                work_dir=Path(td),
                pose_sampler_config=PoseSamplerConfig(
                    n_rotations=1,
                    n_azimuth=1,
                    n_shifts=1,
                    max_poses_per_site=1,
                    min_height=1.8,
                    max_height=2.2,
                    height_step=0.2,
                    random_seed=0,
                ),
                basin_config=BasinConfig(relax_steps=1, energy_window_ev=1.0, desorption_min_bonds=0),
                save_surface_report=False,
                save_site_visualizations=False,
                save_pose_pool=False,
                save_basin_dictionary=False,
                save_basin_ablation=False,
            )
            out = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
        # Regression guard: equivalent top sites should have been compressed before sampling.
        self.assertEqual(int(out.summary["n_primitives"]), int(out.summary["n_basis_primitives"]))


if __name__ == "__main__":
    unittest.main()

