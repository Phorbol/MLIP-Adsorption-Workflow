import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from adsorption_ensemble.pose import PoseSweepConfig, build_slab_cases, list_supported_molecules, run_pose_sampling_sweep, summarize_rows


class TestPoseSweep(unittest.TestCase):
    def test_build_slab_cases_contains_core(self):
        cases = build_slab_cases()
        self.assertIn("fcc111", cases)
        self.assertIn("fcc211", cases)
        self.assertIn("mgo_100_l4", cases)
        self.assertIn("alloy_cuni_111_l4", cases)
        self.assertGreater(len(cases), 10)

    def test_list_supported_molecules_smoke(self):
        names = list_supported_molecules(max_count=20, max_atoms=8)
        self.assertGreater(len(names), 5)
        self.assertTrue(all(isinstance(n, str) and len(n) > 0 for n in names))

    def test_run_pose_sampling_sweep_smoke(self):
        with TemporaryDirectory() as td:
            old = os.environ.get("AE_DISABLE_MACE")
            os.environ["AE_DISABLE_MACE"] = "1"
            try:
                out = run_pose_sampling_sweep(
                    out_root=Path(td),
                    cfg=PoseSweepConfig(
                        n_rotations=2,
                        n_azimuth=8,
                        n_shifts=2,
                        max_basis_sites=3,
                        max_poses_per_site=2,
                        max_poses_output=8,
                        postprocess_enabled=False,
                        random_seed=1,
                    ),
                    max_molecules=2,
                    max_slabs=2,
                    max_atoms_per_molecule=4,
                    max_combinations=3,
                )
            finally:
                if old is None:
                    os.environ.pop("AE_DISABLE_MACE", None)
                else:
                    os.environ["AE_DISABLE_MACE"] = old
            self.assertTrue(Path(out["summary_json"]).exists())
            self.assertTrue(Path(out["summary_csv"]).exists())
            self.assertTrue(Path(out["report_md"]).exists())
            from adsorption_ensemble.workflows import validate_pose_sampling_run

            validate_pose_sampling_run(out)
            pca_plots = list(Path(out["run_dir"]).rglob("site_embedding_pca.png"))
            self.assertGreaterEqual(len(pca_plots), 1)
            self.assertTrue(all(p.exists() and p.stat().st_size > 0 for p in pca_plots))
            self.assertGreater(len(out["rows"]), 0)
            summary = summarize_rows(out["rows"])
            self.assertEqual(summary["total"], len(out["rows"]))
            self.assertGreaterEqual(summary["ok"], 1)
            ok_rows = [r for r in out["rows"] if r.get("ok")]
            self.assertTrue(any(int(r.get("azimuth_unique_n", 0)) >= 2 for r in ok_rows))
            self.assertTrue(all("pose_tilt_median_deg" in r for r in ok_rows))
            self.assertTrue(all("upright_coverage_ok" in r for r in ok_rows))
            self.assertTrue(all("upright_repair_used" in r for r in ok_rows))
            self.assertTrue(all("height_shift_unique_n" in r for r in ok_rows))

    def test_run_pose_sampling_sweep_profiling_fields(self):
        with TemporaryDirectory() as td:
            old = os.environ.get("AE_DISABLE_MACE")
            os.environ["AE_DISABLE_MACE"] = "1"
            try:
                out = run_pose_sampling_sweep(
                    out_root=Path(td),
                    cfg=PoseSweepConfig(
                        n_rotations=1,
                        n_azimuth=2,
                        n_shifts=1,
                        max_basis_sites=1,
                        max_poses_per_site=1,
                        max_poses_output=2,
                        postprocess_enabled=False,
                        profiling_enabled=True,
                        random_seed=0,
                    ),
                    max_molecules=1,
                    max_slabs=1,
                    max_atoms_per_molecule=3,
                    max_combinations=1,
                )
            finally:
                if old is None:
                    os.environ.pop("AE_DISABLE_MACE", None)
                else:
                    os.environ["AE_DISABLE_MACE"] = old
            ok_rows = [r for r in out["rows"] if r.get("ok")]
            self.assertGreaterEqual(len(ok_rows), 1)
            r0 = ok_rows[0]
            self.assertIn("timing_sampling_s", r0)
            self.assertIn("sampler_profile", r0)
            self.assertTrue(isinstance(r0["sampler_profile"], dict))
            prof_file = Path(r0["output_dir"]) / "profiling_pose_sampler.json"
            self.assertTrue(prof_file.exists())

    def test_run_pose_sampling_sweep_ensemble_outputs(self):
        with TemporaryDirectory() as td:
            old = os.environ.get("AE_DISABLE_MACE")
            os.environ["AE_DISABLE_MACE"] = "1"
            try:
                out = run_pose_sampling_sweep(
                    out_root=Path(td),
                    cfg=PoseSweepConfig(
                        n_rotations=2,
                        n_azimuth=6,
                        n_shifts=1,
                        max_basis_sites=2,
                        max_poses_per_site=2,
                        max_poses_output=6,
                        postprocess_enabled=False,
                        ensemble_enabled=True,
                        ensemble_desorption_min_bonds=0,
                        random_seed=2,
                    ),
                    max_molecules=1,
                    max_slabs=1,
                    max_atoms_per_molecule=4,
                    max_combinations=1,
                    slab_filter=["fcc111"],
                )
            finally:
                if old is None:
                    os.environ.pop("AE_DISABLE_MACE", None)
                else:
                    os.environ["AE_DISABLE_MACE"] = old
            ok_rows = [r for r in out["rows"] if r.get("ok")]
            self.assertGreaterEqual(len(ok_rows), 1)
            r0 = ok_rows[0]
            self.assertTrue(bool(r0.get("ensemble_enabled")))
            self.assertGreaterEqual(int(r0.get("ensemble_n_basins", 0)), 1)
            self.assertGreaterEqual(int(r0.get("ensemble_n_nodes", 0)), 1)
            basins_json = Path(r0["output_dir"]) / "basins.json"
            nodes_json = Path(r0["output_dir"]) / "nodes.json"
            self.assertTrue(basins_json.exists() and basins_json.stat().st_size > 0)
            self.assertTrue(nodes_json.exists() and nodes_json.stat().st_size > 0)
            from adsorption_ensemble.workflows import validate_pose_sampling_run

            validate_pose_sampling_run(out)

    def test_mace_config_normalization_and_strict_gate(self):
        from adsorption_ensemble.pose import sweep as sweep_mod

        mp, dev, dt = sweep_mod._normalize_mace_relax_config(model_path=None, device="cuda", dtype="float64", strict=False)
        self.assertEqual(dev, "cuda")
        self.assertEqual(dt, "float32")
        with self.assertRaises(FileNotFoundError):
            sweep_mod._normalize_mace_descriptor_config(model_path="__missing__.model", device="cuda", dtype="float64", strict=True)


if __name__ == "__main__":
    unittest.main()
