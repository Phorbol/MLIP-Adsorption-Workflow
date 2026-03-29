import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.build import molecule

from adsorption_ensemble.conformer_md import (
    ConformerMDSampler,
    ConformerMDSamplerConfig,
    GeometryPairDistanceDescriptor,
    MACEEnergyRelaxBackend,
    MDRunResult,
    XTBMDConfig,
    XTBMDRunner,
    read_molecule_any,
)


class FakeMDRunner:
    def run(self, molecule_atoms, run_dir: Path) -> MDRunResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(12):
            a = molecule_atoms.copy()
            shift = 0.03 * np.sin(0.3 * i + np.arange(len(a))[:, None])
            a.set_positions(a.get_positions() + shift)
            frames.append(a)
        return MDRunResult(frames=frames, metadata={"source": "fake", "n_frames": len(frames)})


class FakeRelaxBackend:
    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        work_dir.mkdir(parents=True, exist_ok=True)
        out = [f.copy() for f in frames]
        energies = np.linspace(0.0, 0.22, len(out), dtype=float)
        return out, energies


class FakeInferencer:
    class _Out:
        def __init__(self, n):
            self.descriptors = np.zeros((n, 4), dtype=float)
            self.energies_per_atom_ev = np.linspace(0.0, 0.1, n, dtype=float)
            self.metadata = {"source": "fake_mace", "n_output_frames": n}

    def infer(self, frames):
        return FakeInferencer._Out(len(frames))


class FakeDescriptorWithMeta:
    def __init__(self):
        self.last_infer_metadata = None

    def transform(self, frames):
        self.last_infer_metadata = {"source": "fake_descriptor", "n_frames": len(frames)}
        return GeometryPairDistanceDescriptor().transform(frames)


class FakeRelaxBackendWithMeta(FakeRelaxBackend):
    def __init__(self):
        self.last_infer_metadata = None

    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        self.last_infer_metadata = {"source": "fake_relax", "n_frames": len(frames), "maxf": maxf, "steps": steps}
        return super().relax_batch(frames, work_dir, maxf=maxf, steps=steps)


class TestConformerMDPipeline(unittest.TestCase):
    def test_xtb_default_md_config_values(self):
        cfg = XTBMDConfig()
        self.assertEqual(cfg.temperature_k, 400.0)
        self.assertEqual(cfg.time_ps, 100.0)
        self.assertEqual(cfg.dump_fs, 50.0)
        self.assertEqual(cfg.step_fs, 1.0)
        self.assertEqual(cfg.hmass, 1)
        self.assertEqual(cfg.shake, 1)

    def test_xtb_md_input_render(self):
        cfg = XTBMDConfig(temperature_k=420.0, time_ps=25.0, step_fs=0.5, dump_fs=20.0, seed=7)
        text = XTBMDRunner(cfg)._render_md_input(include_advanced_keywords=True)
        self.assertIn("$md", text)
        self.assertIn("temp=420.0", text)
        self.assertIn("time=25.0", text)
        self.assertIn("step=0.5", text)
        self.assertIn("seed=7", text)
        self.assertIn("hmass=", text)
        self.assertIn("shake=", text)

    def test_xtb_md_input_render_fallback(self):
        cfg = XTBMDConfig(temperature_k=350.0, time_ps=10.0, step_fs=1.0, dump_fs=10.0, seed=9)
        text = XTBMDRunner(cfg)._render_md_input(include_advanced_keywords=False)
        self.assertIn("temp=350.0", text)
        self.assertIn("seed=9", text)
        self.assertNotIn("hmass=", text)
        self.assertIn("shake=0", text)

    def test_geometry_descriptor_shape(self):
        frames = [molecule("H2O"), molecule("H2O")]
        feats = GeometryPairDistanceDescriptor().transform(frames)
        self.assertEqual(feats.shape[0], 2)
        self.assertEqual(feats.shape[1], 3)

    def test_read_molecule_any_from_gjf(self):
        root = Path(__file__).resolve().parents[1]
        atoms = read_molecule_any(root / "C6.gjf")
        self.assertGreater(len(atoms), 0)

    def test_pipeline_end_to_end_fps(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.preselect_k = 8
        cfg.selection.mode = "fps"
        cfg.selection.energy_window_ev = 0.15
        cfg.selection.rmsd_threshold = 0.02
        cfg.md.n_runs = 2
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=FakeMDRunner(),
                relax_backend=FakeRelaxBackend(),
            )
            result = sampler.run(molecule("H2O"), job_name="ut_fps")
            self.assertGreater(len(result.conformers), 0)
            self.assertLessEqual(np.max(result.energies_ev), 0.15 + 1e-12)
            self.assertTrue((Path(td) / "ut_fps" / "metadata.json").exists())
            self.assertTrue((Path(td) / "ut_fps" / "summary.txt").exists())
            self.assertTrue((Path(td) / "ut_fps" / "summary.json").exists())
            self.assertTrue((Path(td) / "ut_fps" / "stage_metrics.json").exists())
            self.assertTrue((Path(td) / "ut_fps" / "stage_metrics.csv").exists())

    def test_pipeline_end_to_end_fps_with_grid_convergence(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.preselect_k = 8
        cfg.selection.mode = "fps"
        cfg.selection.fps_round_size = 2
        cfg.selection.fps_rounds = 10
        cfg.selection.fps_convergence_enable = True
        cfg.selection.fps_convergence_grid_bins = 2
        cfg.selection.fps_convergence_min_rounds = 2
        cfg.selection.fps_convergence_patience = 1
        cfg.selection.fps_convergence_min_coverage_gain = 1e-6
        cfg.selection.fps_convergence_min_novelty = 0.25
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=FakeMDRunner(),
                relax_backend=FakeRelaxBackend(),
            )
            result = sampler.run(molecule("H2O"), job_name="ut_fps_grid")
            self.assertGreater(len(result.conformers), 0)
            self.assertTrue((Path(td) / "ut_fps_grid" / "summary.json").exists())

    def test_pipeline_end_to_end_kmeans(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.preselect_k = 6
        cfg.selection.mode = "kmeans"
        cfg.selection.energy_window_ev = 0.20
        cfg.selection.rmsd_threshold = 0.03
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=FakeMDRunner(),
                relax_backend=FakeRelaxBackend(),
            )
            result = sampler.run(molecule("CH3OH"), job_name="ut_kmeans")
            self.assertGreater(len(result.conformers), 0)
            self.assertTrue((Path(td) / "ut_kmeans" / "ensemble.extxyz").exists())
            summary = json.loads((Path(td) / "ut_kmeans" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("n_selected", summary)
            self.assertIn("top5_energy_ev", summary)
            stage = json.loads((Path(td) / "ut_kmeans" / "stage_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("counts", stage)
            self.assertIn("retention", stage)
            self.assertIn("energy", stage)
            self.assertIn("diversity", stage)

    def test_invalid_descriptor_backend_raises(self):
        cfg = ConformerMDSamplerConfig()
        cfg.descriptor.backend = "unknown"
        with self.assertRaises(ValueError):
            ConformerMDSampler(config=cfg)

    def test_invalid_relax_backend_raises(self):
        cfg = ConformerMDSamplerConfig()
        cfg.relax.backend = "bad_backend"
        with self.assertRaises(ValueError):
            ConformerMDSampler(config=cfg)

    def test_mace_energy_relax_backend_shape(self):
        backend = MACEEnergyRelaxBackend(inferencer=FakeInferencer())
        frames = [molecule("H2O"), molecule("H2O")]
        out_frames, energies = backend.relax_batch(frames, Path("."))
        self.assertEqual(len(out_frames), 2)
        self.assertEqual(energies.shape, (2,))

    def test_pipeline_metadata_contains_inference_audit(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.preselect_k = 4
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=FakeMDRunner(),
                descriptor_extractor=FakeDescriptorWithMeta(),
                relax_backend=FakeRelaxBackendWithMeta(),
            )
            result = sampler.run(molecule("H2O"), job_name="ut_meta")
            self.assertIn("descriptor_inference", result.metadata)
            self.assertIn("relax_inference", result.metadata)
            self.assertEqual(result.metadata["descriptor_inference"]["source"], "fake_descriptor")
            self.assertEqual(result.metadata["relax_inference"]["source"], "fake_relax")

    def test_read_molecule_any_gaussian_fallback(self):
        gjf_text = "\n".join(
            [
                "%chk=a.chk",
                "# hf/3-21g",
                "",
                "title",
                "",
                "0 1",
                "H 0.0 0.0 0.0",
                "H 0.0 0.0 0.74",
                "",
            ]
        )
        with TemporaryDirectory() as td:
            p = Path(td) / "h2.gjf"
            p.write_text(gjf_text, encoding="utf-8")
            atoms = read_molecule_any(p)
            self.assertEqual(len(atoms), 2)

    def test_xtb_trj_xyz_fallback_reader(self):
        trj_text = "\n".join(
            [
                "2",
                "frame 1",
                "H 0.0 0.0 0.0",
                "H 0.0 0.0 0.74",
                "2",
                "frame 2",
                "H 0.0 0.0 0.01",
                "H 0.0 0.0 0.75",
                "",
            ]
        )
        with TemporaryDirectory() as td:
            trj = Path(td) / "xtb.trj"
            trj.write_text(trj_text, encoding="utf-8")
            frames = XTBMDRunner(XTBMDConfig())._read_md_trajectory(trj)
            if not isinstance(frames, list):
                frames = [frames]
            self.assertEqual(len(frames), 2)
            self.assertEqual(len(frames[0]), 2)

    def test_xtb_can_accept_partial_success_with_trj(self):
        trj_text = "\n".join(
            [
                "2",
                "frame 1",
                "H 0.0 0.0 0.0",
                "H 0.0 0.0 0.74",
                "",
            ]
        )
        with TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "xtb.trj").write_text(trj_text, encoding="utf-8")
            ok = XTBMDRunner(XTBMDConfig())._can_accept_partial_md_success(run_dir)
            self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
