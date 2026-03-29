import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.full_repo_example import run_adsorption_api_example


class TestFullRepoExample(unittest.TestCase):
    def test_run_adsorption_api_example_uses_workflow_api(self):
        with TemporaryDirectory() as td:
            out = run_adsorption_api_example(
                out_root=Path(td),
                use_mace_dedup=False,
                mace_model_path=None,
                mace_device="cpu",
                mace_dtype="float32",
            )
            self.assertGreater(out["n_primitives"], 0)
            self.assertGreater(out["n_poses"], 0)
            self.assertGreaterEqual(out["n_basins"], 1)
            self.assertGreaterEqual(out["n_nodes"], 1)
            self.assertGreater(out["paper_readiness_score"], 0)
            self.assertTrue(Path(out["files"]["basins_json"]).exists())
            self.assertTrue(Path(out["files"]["nodes_json"]).exists())
            self.assertTrue(Path(out["files"]["site_dictionary_json"]).exists())
