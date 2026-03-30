import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ase.build import fcc111, molecule

from adsorption_ensemble.basin import BasinConfig, build_basin_dictionary, run_basin_ablation
from adsorption_ensemble.node import NodeConfig, basin_to_node
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, run_adsorption_workflow


class TestBasinReporting(unittest.TestCase):
    def test_workflow_writes_basin_dictionary_and_ablation(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.6,
                max_height=2.4,
                height_step=0.2,
                max_poses_per_site=2,
                random_seed=0,
            ),
            basin_config=BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=1.0,
                desorption_min_bonds=0,
                work_dir=None,
            ),
            max_primitives=2,
            save_basin_dictionary=True,
            save_basin_ablation=True,
            basin_ablation_metrics=("signature_only", "rmsd"),
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            result = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
            self.assertTrue((Path(td) / "basin_dictionary.json").exists())
            self.assertTrue((Path(td) / "basin_ablation.json").exists())
            self.assertTrue((Path(td) / "raw_site_dictionary.json").exists())
            self.assertTrue((Path(td) / "selected_site_dictionary.json").exists())
            self.assertTrue((Path(td) / "sites.png").exists())
            self.assertTrue((Path(td) / "sites_only.png").exists())
            self.assertTrue((Path(td) / "sites_inequivalent.png").exists())
            self.assertIn("basin_dictionary_json", result.artifacts)
            self.assertIn("basin_ablation_json", result.artifacts)

    def test_basin_dictionary_and_ablation_functions(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.6,
                max_height=2.4,
                height_step=0.2,
                max_poses_per_site=2,
                random_seed=0,
            ),
            basin_config=BasinConfig(relax_maxf=0.1, relax_steps=1, energy_window_ev=1.0, desorption_min_bonds=0, work_dir=None),
            max_primitives=2,
            save_basin_dictionary=False,
            save_basin_ablation=False,
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            result = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
            energy_min = result.basin_result.summary.get("energy_min_ev")
            nodes = [basin_to_node(b, slab_n=len(slab), cfg=NodeConfig(), energy_min_ev=energy_min) for b in result.basin_result.basins]
            basin_dict = build_basin_dictionary(result.basin_result, pose_frames=result.pose_frames, nodes=nodes, slab_n=len(slab))
            self.assertIn("basins", basin_dict)
            self.assertGreaterEqual(len(basin_dict["basins"]), 1)
            ablation = run_basin_ablation(
                frames=result.pose_frames,
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=int(result.surface_context.classification.normal_axis),
                base_config=cfg.basin_config,
                metrics=("signature_only", "rmsd", "mace_node_l2"),
            )
            self.assertIn("metrics", ablation)
            self.assertIn("signature_only", ablation["metrics"])
            self.assertIn("rmsd", ablation["metrics"])
            self.assertIn("mace_node_l2", ablation["metrics"])
            self.assertIn(ablation["metrics"]["mace_node_l2"]["status"], {"ok", "error"})
