import unittest
from unittest.mock import patch

from adsorption_ensemble.conformer_md.config import MACEInferenceConfig
from adsorption_ensemble.conformer_md.mace_inference import MACEBatchInferencer


class _DummyParam:
    def __init__(self):
        self.requires_grad = True


class _DummyModel:
    def __init__(self):
        self._enable_amp = True
        self._params = [_DummyParam()]

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
    def __init__(self, model):
        self._model = model

    def load(self, path, map_location=None):
        return self._model


class TestMACEInference(unittest.TestCase):
    def test_get_model_applies_cueq_conversion_when_enabled(self):
        dummy = _DummyModel()
        infer = MACEBatchInferencer(
            MACEInferenceConfig(
                model_path="/tmp/dummy.model",
                device="cuda",
                dtype="float32",
                enable_cueq=True,
            )
        )
        with patch("mace.calculators.mace.run_e3nn_to_cueq", side_effect=lambda model, device: model) as cueq:
            model = infer._get_model(_FakeTorch(dummy))
        self.assertEqual(cueq.call_count, 1)
        self.assertEqual(cueq.call_args.kwargs["device"], "cuda")
        self.assertFalse(bool(model._enable_amp))
        self.assertTrue(all(param.requires_grad is False for param in model.parameters()))


if __name__ == "__main__":
    unittest.main()
