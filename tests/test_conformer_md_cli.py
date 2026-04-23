import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase.build import molecule

from adsorption_ensemble.conformer_md import cli


class _CaptureSampler:
    last_config = None
    last_job_name = None

    def __init__(self, config):
        type(self).last_config = config

    def run(self, molecule, job_name="conformer_job"):
        type(self).last_job_name = job_name
        return SimpleNamespace(conformers=[], metadata={})


class TestConformerMDCli(unittest.TestCase):
    def test_main_applies_profile_and_explicit_overrides(self):
        with patch("adsorption_ensemble.conformer_md.cli.read_molecule_any", return_value=molecule("H2O")):
            with patch("adsorption_ensemble.conformer_md.cli.ConformerMDSampler", _CaptureSampler):
                with patch("sys.stdout", new=io.StringIO()):
                    rc = cli.main(
                        [
                            "examples/C6H14.gjf",
                            "--job-name",
                            "ut_cli",
                            "--work-dir",
                            "artifacts/ut_cli",
                            "--generator-backend",
                            "rdkit_embed",
                            "--selection-profile",
                            "adsorption_seed_broad",
                            "--energy-semantics",
                            "per_atom",
                            "--pair-energy-gap-ev",
                            "0.25",
                            "--loose-filter",
                            "energy",
                            "--final-filter",
                            "rmsd",
                            "--structure-metric-threshold",
                            "0.11",
                            "--loose-structure-metric-threshold",
                            "0.21",
                            "--final-structure-metric-threshold",
                            "0.31",
                            "--rdkit-num-confs",
                            "48",
                            "--rdkit-prune-rms-thresh",
                            "0.15",
                            "--rdkit-optimize-forcefield",
                            "uff",
                        ]
                    )
        self.assertEqual(rc, 0)
        cfg = _CaptureSampler.last_config
        self.assertIsNotNone(cfg)
        self.assertEqual(_CaptureSampler.last_job_name, "ut_cli")
        self.assertEqual(cfg.generator.backend, "rdkit_embed")
        self.assertEqual(cfg.selection.selection_profile, "adsorption_seed_broad")
        self.assertEqual(cfg.selection.target_final_k, 8)
        self.assertFalse(cfg.selection.use_total_energy)
        self.assertAlmostEqual(cfg.selection.pair_energy_gap_ev, 0.25)
        self.assertEqual(cfg.selection.loose_filter, "energy")
        self.assertEqual(cfg.selection.final_filter, "rmsd")
        self.assertAlmostEqual(cfg.selection.rmsd_threshold, 0.11)
        self.assertAlmostEqual(cfg.selection.loose_rmsd_threshold, 0.21)
        self.assertAlmostEqual(cfg.selection.final_rmsd_threshold, 0.31)
        self.assertEqual(cfg.generator.rdkit.num_confs, 48)
        self.assertAlmostEqual(cfg.generator.rdkit.prune_rms_thresh, 0.15)
        self.assertEqual(cfg.generator.rdkit.optimize_forcefield, "uff")
        self.assertEqual(cfg.output.work_dir, Path("artifacts/ut_cli"))


if __name__ == "__main__":
    unittest.main()
