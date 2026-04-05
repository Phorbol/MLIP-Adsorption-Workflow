import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.build import fcc111, molecule

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.conformer_md import ConformerMDSamplerConfig, GeometryPairDistanceDescriptor
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.workflows import (
    AdsorptionWorkflowConfig,
    evaluate_adsorption_workflow_readiness,
    run_adsorption_workflow,
)


class FakeMDRunner:
    def run(self, molecule_atoms, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(10):
            a = molecule_atoms.copy()
            shift = 0.02 * np.sin(0.5 * i + np.arange(len(a))[:, None])
            a.set_positions(a.get_positions() + shift)
            frames.append(a)
        return type("MDRunResult", (), {"frames": frames, "metadata": {"source": "fake", "n_frames": len(frames)}})()


class FakeRelaxBackend:
    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        work_dir.mkdir(parents=True, exist_ok=True)
        return [f.copy() for f in frames], np.linspace(0.0, 0.15, len(frames), dtype=float)


class TestAdsorptionWorkflow(unittest.TestCase):
    def test_run_adsorption_workflow_rigid_smoke(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=2,
                n_azimuth=4,
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
                relax_steps=2,
                energy_window_ev=1.0,
                desorption_min_bonds=0,
                work_dir=None,
            ),
            max_primitives=3,
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            result = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
            self.assertGreater(result.summary["n_primitives"], 0)
            self.assertGreater(result.summary["n_pose_frames"], 0)
            self.assertGreaterEqual(result.summary["n_basins"], 1)
            self.assertGreaterEqual(result.summary["n_nodes"], 1)
            self.assertIn("surface_diagnostics", result.summary)
            self.assertIn("surface_classification", result.summary)
            self.assertEqual(result.summary["surface_classification"]["normal_axis"], 2)
            self.assertTrue((Path(td) / "site_dictionary.json").exists())
            self.assertTrue((Path(td) / "basins.json").exists())
            self.assertTrue((Path(td) / "nodes.json").exists())
            report = evaluate_adsorption_workflow_readiness(result)
            self.assertEqual(report.score, report.max_score)

    def test_run_adsorption_workflow_flexible_with_fake_conformer_search(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CH3OH")
        conformer_cfg = ConformerMDSamplerConfig()
        conformer_cfg.selection.preselect_k = 4
        conformer_cfg.selection.mode = "fps"
        conformer_cfg.output.work_dir = Path("unused")
        cfg = AdsorptionWorkflowConfig(
            run_conformer_search=True,
            conformer_config=conformer_cfg,
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.6,
                max_height=2.6,
                height_step=0.2,
                max_poses_per_site=1,
                random_seed=1,
            ),
            basin_config=BasinConfig(
                relax_maxf=0.1,
                relax_steps=2,
                energy_window_ev=2.0,
                desorption_min_bonds=0,
                work_dir=None,
            ),
            max_primitives=2,
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            cfg.conformer_config.output.work_dir = Path(td) / "conformer_runs"
            result = run_adsorption_workflow(
                slab=slab,
                adsorbate=ads,
                config=cfg,
                md_runner=FakeMDRunner(),
                conformer_descriptor_extractor=GeometryPairDistanceDescriptor(),
                conformer_relax_backend=FakeRelaxBackend(),
            )
            self.assertTrue(result.summary["run_conformer_search"])
            self.assertGreaterEqual(result.summary["n_conformers"], 1)
            self.assertTrue((Path(td) / "conformer_metadata.json").exists())
            self.assertTrue((Path(td) / "workflow_summary.json").exists())
