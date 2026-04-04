import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from ase import Atoms

from adsorption_ensemble.relax.backends import (
    MACEBatchRelaxBackend,
    MaceRelaxConfig,
    normalize_mace_descriptor_config,
)


class _DummyCalc:
    def __init__(self, device: str = "cuda"):
        self.device = device


class _DummyBatchRelaxer:
    def __init__(self, calculator, max_edges_per_batch: int, device: str):
        self.calculator = calculator
        self.max_edges_per_batch = int(max_edges_per_batch)
        self.device = str(device)

    def relax(
        self,
        atoms_list,
        fmax,
        head,
        max_n_steps,
        inplace,
        trajectory_dir,
        append_trajectory_file,
        save_log_file,
    ):
        return [atoms.copy() for atoms in atoms_list]


class TestRelaxBackends(unittest.TestCase):
    def test_normalize_mace_descriptor_config_falls_back_to_cpu_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            model_path, device, dtype = normalize_mace_descriptor_config(
                model_path="/root/.cache/mace/mace-omat-0-small.model",
                device="cuda",
                dtype="fp32",
                strict=False,
            )
        self.assertEqual(model_path, "/root/.cache/mace/mace-omat-0-small.model")
        self.assertEqual(device, "cpu")
        self.assertEqual(dtype, "float32")

    def test_batch_relax_backend_forwards_enable_cueq_and_reports_backend_flag(self):
        backend = MACEBatchRelaxBackend(
            MaceRelaxConfig(
                model_path="/root/.cache/mace/mace-omat-0-small.model",
                device="cuda",
                dtype="float32",
                enable_cueq=True,
                strict=False,
            )
        )
        frame = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        with TemporaryDirectory() as td:
            with (
                patch(
                    "adsorption_ensemble.relax.backends.normalize_mace_relax_config",
                    return_value=("/root/.cache/mace/mace-omat-0-small.model", "cuda", "float32"),
                ),
                patch(
                    "adsorption_ensemble.relax.backends.get_mace_calc",
                    return_value=_DummyCalc(device="cuda"),
                ) as get_calc,
                patch("adsorption_ensemble.conformer_md.mace_batch_relax.BatchRelaxer", _DummyBatchRelaxer),
            ):
                out_frames, energies, backend_name = backend.relax(
                    frames=[frame],
                    maxf=0.1,
                    steps=2,
                    work_dir=Path(td),
                )
        self.assertEqual(len(out_frames), 1)
        self.assertEqual(energies.shape, (1,))
        self.assertTrue(np.isnan(float(energies[0])))
        self.assertIn("cueq=1", backend_name)
        self.assertTrue(bool(get_calc.call_args.kwargs["enable_cueq"]))


if __name__ == "__main__":
    unittest.main()
