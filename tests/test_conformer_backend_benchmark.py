import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ase.build import molecule

from tools import run_conformer_backend_benchmark


class _FakeSampler:
    def __init__(self, config):
        self.config = config

    def run(self, molecule, job_name="conformer_job"):
        backend = str(self.config.generator.backend)
        return SimpleNamespace(
            conformers=[molecule.copy()],
            metadata={
                "n_raw_frames": 10 if backend == "xtb_md" else 7,
                "n_preselected": 4,
                "n_after_loose_filter": 3,
                "n_selected": 2,
                "generator_summary": {"walltime_generation_s": 0.5 if backend == "xtb_md" else 0.2},
                "result_summary": {
                    "energy_min_ev": -1.0,
                    "energy_mean_ev": -0.8,
                },
            },
        )


class TestConformerBackendBenchmark(unittest.TestCase):
    def test_benchmark_writes_summary_for_both_backends(self):
        with TemporaryDirectory() as td:
            with patch("tools.run_conformer_backend_benchmark.read_molecule_any", return_value=molecule("H2O")):
                with patch("tools.run_conformer_backend_benchmark.ConformerMDSampler", _FakeSampler):
                    with patch("sys.stdout", new=io.StringIO()):
                        rc = run_conformer_backend_benchmark.main(
                            [
                                "examples/C6H14.gjf",
                                "--out-root",
                                td,
                            ]
                        )
            self.assertEqual(rc, 0)
            summary_path = Path(td) / "benchmark_summary.json"
            self.assertTrue(summary_path.exists())
            rows = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["backend"] for row in rows}, {"xtb_md", "rdkit_embed"})
            self.assertTrue(all(row["status"] == "ok" for row in rows))


if __name__ == "__main__":
    unittest.main()
