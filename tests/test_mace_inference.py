import unittest
from unittest.mock import patch
import types
import sys

from adsorption_ensemble.conformer_md.config import MACEInferenceConfig
from adsorption_ensemble.conformer_md.mace_inference import MACEBatchInferencer


class _DummyParam:
    def __init__(self):
        self.requires_grad = True


class _DummyModel:
    def __init__(self):
        self._enable_amp = True
        self._params = [_DummyParam()]
        self.head_name = "matpes_r2scan"
        self.head_names = ["matpes_r2scan", "omol"]

    def float(self):
        return self

    def double(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(self._params)


class _FakeTorch:
    class _Cuda:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return bool(self._available)

    def __init__(self, model, *, cuda_available=True):
        self._model = model
        self.cuda = self._Cuda(cuda_available)
        self.last_map_location = None

    def load(self, path, map_location=None):
        self.last_map_location = map_location
        return self._model


class TestMACEInference(unittest.TestCase):
    def test_resolve_head_name_prefers_explicit_config(self):
        infer = MACEBatchInferencer(
            MACEInferenceConfig(
                model_path="/tmp/dummy.model",
                head_name="omol",
            )
        )
        self.assertEqual(infer._resolve_head_name(_DummyModel()), "omol")

    def test_resolve_head_name_rejects_unknown_explicit_head(self):
        infer = MACEBatchInferencer(
            MACEInferenceConfig(
                model_path="/tmp/dummy.model",
                head_name="unknown_head",
            )
        )
        with self.assertRaises(ValueError):
            infer._resolve_head_name(_DummyModel())

    def test_get_model_applies_cueq_conversion_when_enabled(self):
        dummy = _DummyModel()
        fake_torch = _FakeTorch(dummy, cuda_available=True)
        infer = MACEBatchInferencer(
            MACEInferenceConfig(
                model_path="/tmp/dummy.model",
                device="cuda",
                dtype="float32",
                enable_cueq=True,
            )
        )
        fake_mace = types.ModuleType("mace")
        fake_calculators = types.ModuleType("mace.calculators")
        fake_calculators_mace = types.ModuleType("mace.calculators.mace")
        fake_calculators_mace.run_e3nn_to_cueq = lambda model, device: model
        with patch.dict(
            sys.modules,
            {
                "mace": fake_mace,
                "mace.calculators": fake_calculators,
                "mace.calculators.mace": fake_calculators_mace,
            },
        ):
            with patch("mace.calculators.mace.run_e3nn_to_cueq", side_effect=lambda model, device: model) as cueq:
                runtime_device, _ = infer._resolve_runtime_device(fake_torch)
                model = infer._get_model(fake_torch, runtime_device=runtime_device)
        self.assertEqual(cueq.call_count, 1)
        self.assertEqual(cueq.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(fake_torch.last_map_location, "cuda:0")
        self.assertFalse(bool(model._enable_amp))
        self.assertTrue(all(param.requires_grad is False for param in model.parameters()))

    def test_get_model_falls_back_to_cpu_when_cuda_unavailable(self):
        dummy = _DummyModel()
        fake_torch = _FakeTorch(dummy, cuda_available=False)
        infer = MACEBatchInferencer(
            MACEInferenceConfig(
                model_path="/tmp/dummy.model",
                device="cuda",
                dtype="float32",
                enable_cueq=True,
            )
        )
        runtime_device, runtime_meta = infer._resolve_runtime_device(fake_torch)
        model = infer._get_model(fake_torch, runtime_device=runtime_device)
        self.assertEqual(runtime_device, "cpu")
        self.assertEqual(runtime_meta["device"], "cpu")
        self.assertEqual(fake_torch.last_map_location, "cpu")
        self.assertEqual(getattr(model, "device", None), "cpu")
        self.assertTrue(all(param.requires_grad is False for param in model.parameters()))


if __name__ == "__main__":
    unittest.main()
