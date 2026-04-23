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
    resolve_selection_profile,
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


class FakeEnergyAwareDescriptor:
    def __init__(self):
        self.last_infer_metadata = None
        self.last_energies_ev = None

    def transform(self, frames):
        self.last_infer_metadata = {"source": "fake_energy_descriptor", "n_frames": len(frames)}
        self.last_energies_ev = np.linspace(0.5, 0.0, len(frames), dtype=float)
        feats = np.zeros((len(frames), 2), dtype=float)
        for i in range(len(frames)):
            feats[i, 0] = 0.0 if i < len(frames) // 2 else 10.0
            feats[i, 1] = float(i)
        return feats


class FakeRelaxBackendWithMeta(FakeRelaxBackend):
    def __init__(self):
        self.last_infer_metadata = None

    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        self.last_infer_metadata = {"source": "fake_relax", "n_frames": len(frames), "maxf": maxf, "steps": steps}
        return super().relax_batch(frames, work_dir, maxf=maxf, steps=steps)


class SeedAwareMDRunner:
    def __init__(self, seed: int = 42):
        self.config = XTBMDConfig(seed=seed)
        self.seen_seeds: list[int] = []

    def run(self, molecule_atoms, run_dir: Path) -> MDRunResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.seen_seeds.append(int(self.config.seed))
        frames = [molecule_atoms.copy()]
        return MDRunResult(frames=frames, metadata={"seed": int(self.config.seed), "n_frames": 1})


class IndexedFeatureDescriptor:
    def __init__(self, features: np.ndarray, energies: np.ndarray | None = None):
        self.features = np.asarray(features, dtype=float)
        self.energies = None if energies is None else np.asarray(energies, dtype=float)
        self.last_infer_metadata = None
        self.last_energies_ev = None

    def transform(self, frames):
        n = len(frames)
        self.last_infer_metadata = {"source": "indexed_descriptor", "n_frames": n}
        if self.energies is not None and len(self.energies) == n:
            self.last_energies_ev = np.asarray(self.energies, dtype=float)
        else:
            self.last_energies_ev = None
        return np.asarray(self.features[:n], dtype=float)


class IndexedRelaxBackend:
    def __init__(self, energies: np.ndarray):
        self.energies = np.asarray(energies, dtype=float)

    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        work_dir.mkdir(parents=True, exist_ok=True)
        out = [f.copy() for f in frames]
        energies = []
        for i, atoms in enumerate(out):
            frame_id = int(atoms.info.get("frame_id", i))
            energies.append(float(self.energies[frame_id]))
        return out, np.asarray(energies, dtype=float)


def _conformer_test_frames() -> list:
    base = molecule("H2O")
    frames = []
    a0 = base.copy()
    a0.info["frame_id"] = 0
    frames.append(a0)
    a1 = base.copy()
    p1 = a1.get_positions()
    p1[1, 0] += 0.01
    a1.set_positions(p1)
    a1.info["frame_id"] = 1
    frames.append(a1)
    a2 = base.copy()
    p2 = a2.get_positions()
    p2[1, 0] += 0.60
    a2.set_positions(p2)
    a2.info["frame_id"] = 2
    frames.append(a2)
    a3 = base.copy()
    p3 = a3.get_positions()
    p3[1, 0] += 0.61
    a3.set_positions(p3)
    a3.info["frame_id"] = 3
    frames.append(a3)
    return frames


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

    def test_pipeline_applies_explicit_target_final_k(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.preselect_k = 8
        cfg.selection.target_final_k = 3
        cfg.selection.energy_window_ev = 1.0
        cfg.selection.rmsd_threshold = 0.0
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=FakeMDRunner(),
                relax_backend=FakeRelaxBackend(),
            )
            result = sampler.run(molecule("CH3OH"), job_name="ut_target_final_k")
            self.assertEqual(len(result.conformers), 3)
            self.assertEqual(result.metadata["result_summary"]["n_selected"], 3)
            self.assertEqual(result.metadata["selection_profile"], "manual")

    def test_pipeline_uses_descriptor_supplied_raw_energies_for_preselection(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.preselect_k = 2
        cfg.selection.mode = "kmeans"
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=FakeMDRunner(),
                descriptor_extractor=FakeEnergyAwareDescriptor(),
                relax_backend=FakeRelaxBackend(),
            )
            result = sampler.run(molecule("H2O"), job_name="ut_raw_energy_preselect")
            pre_ids = result.metadata["preselected_ids_in_raw"]
            self.assertEqual(len(pre_ids), 2)
            self.assertIn(5, pre_ids)
            self.assertIn(11, pre_ids)

    def test_selection_profile_adsorption_seed_broad_prefers_mace_fp64(self):
        cfg = ConformerMDSamplerConfig()
        cfg.descriptor.mace.model_path = "/tmp/fake.model"
        out = resolve_selection_profile(cfg, profile="adsorption_seed_broad")
        self.assertEqual(out.selection.selection_profile, "adsorption_seed_broad")
        self.assertEqual(out.selection.target_final_k, 8)
        self.assertEqual(out.selection.energy_window_ev, 0.60)
        self.assertEqual(out.selection.metric_backend, "mace")
        self.assertEqual(out.descriptor.backend, "mace")
        self.assertEqual(out.descriptor.mace.dtype, "float64")
        self.assertFalse(out.descriptor.mace.enable_cueq)

    def test_sampler_diversifies_seed_per_md_run(self):
        cfg = ConformerMDSamplerConfig()
        cfg.md.seed = 42
        cfg.md.n_runs = 3
        cfg.md.seed_mode = "increment_per_run"
        cfg.selection.preselect_k = 1
        runner = SeedAwareMDRunner(seed=42)
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                md_runner=runner,
                relax_backend=FakeRelaxBackend(),
            )
            result = sampler.run(molecule("H2O"), job_name="ut_seed_mode")
            self.assertEqual(runner.seen_seeds, [42, 43, 44])
            self.assertEqual(result.metadata["per_run_seeds"], [42, 43, 44])

    def test_structure_metric_threshold_alias_tracks_rmsd_threshold(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.rmsd_threshold = 0.07
        self.assertAlmostEqual(cfg.selection.structure_metric_threshold, 0.07)
        cfg.selection.structure_metric_threshold = 0.12
        self.assertAlmostEqual(cfg.selection.rmsd_threshold, 0.12)

    def test_pair_energy_gap_keeps_near_duplicates_when_energy_separated(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.mode = "fps"
        cfg.selection.preselect_k = 4
        cfg.selection.target_final_k = None
        cfg.selection.use_total_energy = False
        cfg.selection.structure_metric_threshold = 0.05
        cfg.selection.energy_window_ev = 1.0
        cfg.selection.pair_energy_gap_ev = 0.20
        frames = _conformer_test_frames()
        raw_features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.01],
                [1.0, 0.0],
                [1.0, 0.01],
            ],
            dtype=float,
        )
        relax_energies = np.asarray([0.0, 0.30, 0.02, 0.40], dtype=float)
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                descriptor_extractor=IndexedFeatureDescriptor(raw_features),
                relax_backend=IndexedRelaxBackend(relax_energies),
            )
            result = sampler.run_from_frames(
                frames=frames,
                job_name="ut_pair_energy_gap",
                raw_features=raw_features,
            )
            self.assertEqual(len(result.conformers), 4)

    def test_zero_pair_energy_gap_preserves_strict_metric_dedup(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.mode = "fps"
        cfg.selection.preselect_k = 4
        cfg.selection.target_final_k = None
        cfg.selection.structure_metric_threshold = 0.05
        cfg.selection.energy_window_ev = 1.0
        cfg.selection.pair_energy_gap_ev = 0.0
        frames = _conformer_test_frames()
        raw_features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.01],
                [1.0, 0.0],
                [1.0, 0.01],
            ],
            dtype=float,
        )
        relax_energies = np.asarray([0.0, 0.30, 0.02, 0.40], dtype=float)
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                descriptor_extractor=IndexedFeatureDescriptor(raw_features),
                relax_backend=IndexedRelaxBackend(relax_energies),
            )
            result = sampler.run_from_frames(
                frames=frames,
                job_name="ut_pair_energy_gap_zero",
                raw_features=raw_features,
            )
            self.assertEqual(len(result.conformers), 2)

    def test_pipeline_uses_total_energy_semantics_by_default(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.mode = "fps"
        cfg.selection.preselect_k = 2
        cfg.selection.target_final_k = None
        cfg.selection.energy_window_ev = 1.0
        cfg.selection.structure_metric_threshold = 0.0
        frames = _conformer_test_frames()[:2]
        raw_features = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        relax_energies = np.asarray([1.0, 1.1], dtype=float)
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                descriptor_extractor=IndexedFeatureDescriptor(raw_features, energies=np.asarray([0.2, 0.3], dtype=float)),
                relax_backend=IndexedRelaxBackend(relax_energies),
            )
            result = sampler.run_from_frames(
                frames=frames,
                job_name="ut_total_energy",
                raw_features=raw_features,
            )
            self.assertEqual(result.metadata["energy_semantics"], "total_ev")
            self.assertTrue(np.allclose(result.energies_ev, np.asarray([3.0, 3.3], dtype=float)))
            self.assertAlmostEqual(float(result.metadata["result_summary"]["energy_min_ev"]), 3.0)

    def test_pipeline_can_preserve_per_atom_energy_semantics(self):
        cfg = ConformerMDSamplerConfig()
        cfg.selection.mode = "fps"
        cfg.selection.preselect_k = 2
        cfg.selection.target_final_k = None
        cfg.selection.energy_window_ev = 1.0
        cfg.selection.structure_metric_threshold = 0.0
        cfg.selection.use_total_energy = False
        frames = _conformer_test_frames()[:2]
        raw_features = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        relax_energies = np.asarray([1.0, 1.1], dtype=float)
        with TemporaryDirectory() as td:
            cfg.output.work_dir = Path(td)
            sampler = ConformerMDSampler(
                config=cfg,
                descriptor_extractor=IndexedFeatureDescriptor(raw_features, energies=np.asarray([0.2, 0.3], dtype=float)),
                relax_backend=IndexedRelaxBackend(relax_energies),
            )
            result = sampler.run_from_frames(
                frames=frames,
                job_name="ut_per_atom_energy",
                raw_features=raw_features,
            )
            self.assertEqual(result.metadata["energy_semantics"], "per_atom_ev")
            self.assertTrue(np.allclose(result.energies_ev, np.asarray([1.0, 1.1], dtype=float)))

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

    def test_read_molecule_any_gaussian_preserves_connectivity_info(self):
        gjf_text = "\n".join(
            [
                "%chk=a.chk",
                "# hf/3-21g geom=connectivity",
                "",
                "title",
                "",
                "0 1",
                "C 0.0 0.0 0.0",
                "H 0.0 0.0 1.0",
                "",
                "1 2 1.0",
                "2",
                "",
            ]
        )
        with TemporaryDirectory() as td:
            p = Path(td) / "ch.gjf"
            p.write_text(gjf_text, encoding="utf-8")
            atoms = read_molecule_any(p)
            self.assertEqual(int(atoms.info["gaussian_charge"]), 0)
            self.assertEqual(int(atoms.info["gaussian_multiplicity"]), 1)
            self.assertEqual(tuple(atoms.info["connectivity_bonds"]), ((0, 1, 1.0),))

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
