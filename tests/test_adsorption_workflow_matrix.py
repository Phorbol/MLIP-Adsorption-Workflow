import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ase.build import bcc110, bulk, fcc100, fcc111, fcc211, surface

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, evaluate_adsorption_workflow_readiness, run_adsorption_workflow

from tests.chemistry_cases import get_test_adsorbate_cases


class TestAdsorptionWorkflowMatrix(unittest.TestCase):
    def _cfg(self, work_dir: Path) -> AdsorptionWorkflowConfig:
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.5,
                max_height=2.5,
                height_step=0.25,
                max_poses_per_site=1,
                random_seed=0,
            ),
            basin_config=BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=2.0,
                desorption_min_bonds=0,
                work_dir=None,
            ),
            max_primitives=2,
        )
        cfg.work_dir = work_dir
        return cfg

    def test_surface_adsorbate_matrix_smoke(self):
        adsorbates = get_test_adsorbate_cases()
        slabs = {
            "fcc111": fcc111("Pt", size=(3, 3, 3), vacuum=10.0),
            "fcc100": fcc100("Pt", size=(3, 3, 3), vacuum=10.0),
            "fcc211": fcc211("Pt", size=(6, 3, 3), vacuum=10.0),
            "bcc110": bcc110("Fe", size=(3, 3, 3), vacuum=10.0),
            "cu321": surface(bulk("Cu", "fcc", a=3.6, cubic=True), (3, 2, 1), layers=3, vacuum=10.0).repeat((2, 1, 1)),
        }
        selected_cases = [
            ("fcc111", "CO"),
            ("fcc100", "H2O"),
            ("fcc211", "CH3OH"),
            ("bcc110", "C2H4"),
            ("cu321", "C6H6"),
            ("fcc111", "glucose_chain_like"),
            ("fcc100", "glucose_ring_like"),
            ("fcc211", "glycine_like"),
            ("bcc110", "dipeptide_like"),
            ("cu321", "p_nitrochlorobenzene_like"),
            ("fcc111", "p_nitrobenzoic_acid_like"),
            ("fcc100", "C2H6"),
            ("fcc211", "C2H2"),
        ]
        with TemporaryDirectory() as td:
            for slab_name, ads_name in selected_cases:
                with self.subTest(slab=slab_name, adsorbate=ads_name):
                    work_dir = Path(td) / slab_name / ads_name
                    result = run_adsorption_workflow(
                        slab=slabs[slab_name],
                        adsorbate=adsorbates[ads_name],
                        config=self._cfg(work_dir),
                    )
                    report = evaluate_adsorption_workflow_readiness(result)
                    self.assertGreater(result.summary["n_primitives"], 0)
                    self.assertGreater(result.summary["n_pose_frames"], 0)
                    self.assertGreaterEqual(report.score, report.max_score - 1)
                    self.assertTrue((work_dir / "workflow_summary.json").exists())
