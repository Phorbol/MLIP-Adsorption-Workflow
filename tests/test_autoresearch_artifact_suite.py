import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_autoresearch_artifact_suite import resolve_mace_model_path, runtime_manifest


class TestAutoresearchArtifactSuiteHelpers(unittest.TestCase):
    def test_resolve_mace_model_path_prefers_cli(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "x.model"
            p.write_text("stub", encoding="utf-8")
            model, src = resolve_mace_model_path(str(p))
            self.assertEqual(model, str(p))
            self.assertEqual(src, "cli")

    def test_resolve_mace_model_path_uses_env(self):
        old = os.environ.get("AE_MACE_MODEL_PATH")
        try:
            with TemporaryDirectory() as td:
                p = Path(td) / "x.model"
                p.write_text("stub", encoding="utf-8")
                os.environ["AE_MACE_MODEL_PATH"] = str(p)
                model, src = resolve_mace_model_path("")
                self.assertEqual(model, str(p))
                self.assertEqual(src, "env")
        finally:
            if old is None:
                os.environ.pop("AE_MACE_MODEL_PATH", None)
            else:
                os.environ["AE_MACE_MODEL_PATH"] = old

    def test_runtime_manifest_written(self):
        with TemporaryDirectory() as td:
            out_root = Path(td)
            payload = runtime_manifest(
                mace_model_path=None,
                model_source="none",
                mace_device="cuda",
                mace_device_effective="cpu",
                mace_dtype="float32",
                out_root=out_root,
            )
            self.assertTrue((out_root / "runtime_manifest.json").exists())
            self.assertIn("python_executable", payload)
            self.assertIn("mace_model_source", payload)
            self.assertIn("mace_device_effective", payload)
