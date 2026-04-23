import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.build import molecule

from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, RDKitConformerGenerator
from adsorption_ensemble.conformer_md.config import RDKitEmbedConfig
from adsorption_ensemble.conformer_md.xtb import MDRunResult, XTBMDConfig


class _FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, molecule_atoms, run_dir: Path) -> MDRunResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.calls += 1
        frames = [molecule_atoms.copy(), molecule_atoms.copy()]
        for i, atoms in enumerate(frames):
            atoms.info["generator_energy_ev"] = float(0.1 * (i + 1))
        return MDRunResult(
            frames=frames,
            metadata={
                "generator_backend": "rdkit_embed",
                "n_requested_confs": 2,
                "n_embedded_confs": 2,
                "graph_source": "stub",
            },
        )


class _IdentityRelax:
    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        work_dir.mkdir(parents=True, exist_ok=True)
        return [f.copy() for f in frames], np.zeros(len(frames), dtype=float)


class TestConformerGeneratorBackends(unittest.TestCase):
    def test_config_md_alias_updates_generator_xtb_config(self):
        cfg = ConformerMDSamplerConfig()
        cfg.md = XTBMDConfig(seed=17)
        self.assertEqual(cfg.generator.xtb.seed, 17)
        cfg.md.temperature_k = 555.0
        self.assertAlmostEqual(cfg.generator.xtb.temperature_k, 555.0)

    def test_rdkit_generator_reports_missing_dependency_cleanly(self):
        gen = RDKitConformerGenerator(RDKitEmbedConfig())
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate(molecule("H2O"), Path("unused"))
        self.assertIn("RDKit", str(ctx.exception))

    def test_rdkit_bond_order_heuristic_keeps_nitro_no_non_aromatic(self):
        atoms = Atoms(symbols=["C", "C", "N", "O"], positions=np.zeros((4, 3), dtype=float))
        self.assertTrue(RDKitConformerGenerator._use_aromatic_bond(atoms, 0, 1))
        self.assertFalse(RDKitConformerGenerator._use_aromatic_bond(atoms, 2, 3))

    def test_pipeline_accepts_rdkit_backend_with_injected_generator(self):
        cfg = ConformerMDSamplerConfig()
        cfg.generator.backend = "rdkit_embed"
        cfg.selection.mode = "fps"
        cfg.selection.preselect_k = 2
        cfg.selection.target_final_k = 1
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            fake = _FakeGenerator()
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=fake,
                relax_backend=_IdentityRelax(),
            )
            result = sampler.run(molecule("H2O"), job_name="ut_rdkit_backend")
            self.assertEqual(fake.calls, 1)
            self.assertEqual(result.metadata["generator_backend"], "rdkit_embed")
            self.assertEqual(result.metadata["raw_energy_source"], "frame_info")
            self.assertEqual(result.metadata["md_runs"], [])
            self.assertEqual(len(result.metadata["generator_runs"]), 1)
            self.assertTrue((Path(td) / "ut_rdkit_backend" / "raw_generated.extxyz").exists())


if __name__ == "__main__":
    unittest.main()
