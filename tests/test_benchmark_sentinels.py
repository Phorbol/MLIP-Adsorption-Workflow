import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.io import write

from adsorption_ensemble.benchmark import audit_cu111_co_case, summarize_adsorbate_binding_environment


def _linear_co_at(position) -> Atoms:
    x, y, z = position
    return Atoms(
        "CO",
        positions=[
            [float(x), float(y), float(z)],
            [float(x), float(y), float(z + 1.15)],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[True, True, False],
    )


class TestBenchmarkSentinels(unittest.TestCase):
    def test_binding_environment_distinguishes_atop_and_hollow(self):
        slab = fcc111("Cu", size=(3, 3, 3), vacuum=12.0)
        top_ids = [i for i, p in enumerate(slab.get_positions()) if abs(float(p[2]) - float(slab.positions[:, 2].max())) <= 1.0e-6]
        top_pos = slab.positions[top_ids]
        ref = top_pos[0]
        nn_order = np.argsort(np.linalg.norm(top_pos - ref[None, :], axis=1))
        top_a = top_pos[int(nn_order[0])]
        top_b = top_pos[int(nn_order[1])]
        top_c = top_pos[int(nn_order[2])]

        atop = slab.copy() + _linear_co_at([top_a[0], top_a[1], float(top_a[2] + 1.85)])
        hollow_xy = (top_a[:2] + top_b[:2] + top_c[:2]) / 3.0
        hollow = slab.copy() + _linear_co_at([float(hollow_xy[0]), float(hollow_xy[1]), float(top_a[2] + 1.30)])

        atop_env = summarize_adsorbate_binding_environment(atop, slab_n=len(slab), ads_atom_index=0, binding_tau=1.15)
        hollow_env = summarize_adsorbate_binding_environment(hollow, slab_n=len(slab), ads_atom_index=0, binding_tau=1.15)
        self.assertEqual(atop_env["coordination"], 1)
        self.assertGreaterEqual(hollow_env["coordination"], 3)

    def test_cu111_co_case_audit_flags_suspicious_hollow_collapse(self):
        slab = fcc111("Cu", size=(3, 3, 3), vacuum=12.0)
        top_ids = [i for i, p in enumerate(slab.get_positions()) if abs(float(p[2]) - float(slab.positions[:, 2].max())) <= 1.0e-6]
        top_pos = slab.positions[top_ids]
        ref = top_pos[0]
        nn_order = np.argsort(np.linalg.norm(top_pos - ref[None, :], axis=1))
        top_a = top_pos[int(nn_order[0])]
        top_b = top_pos[int(nn_order[1])]
        top_c = top_pos[int(nn_order[2])]
        hollow_xy = (top_a[:2] + top_b[:2] + top_c[:2]) / 3.0
        frame = slab.copy() + _linear_co_at([float(hollow_xy[0]), float(hollow_xy[1]), float(top_a[2] + 1.30)])

        with TemporaryDirectory() as td:
            case_dir = Path(td)
            ours = case_dir / "ours"
            ours.mkdir(parents=True, exist_ok=True)
            (ours / "selected_site_dictionary.json").write_text(
                json.dumps({"sites": [{"site_label": "ontop"}, {"site_label": "bridge"}, {"site_label": "fcc"}, {"site_label": "hcp"}]}),
                encoding="utf-8",
            )
            (ours / "basins.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "n_basins": 1,
                            "final_basin_merge": {
                                "enabled": True,
                                "metric": "pure_mace",
                                "n_input_basins": 4,
                                "n_output_basins": 1,
                            },
                        },
                        "basins": [{"basin_id": 0, "signature": "sig0"}],
                    }
                ),
                encoding="utf-8",
            )
            write((ours / "basins.extxyz").as_posix(), [frame])
            payload = audit_cu111_co_case(case_dir)
        self.assertEqual(payload["n_basis_sites"], 4)
        self.assertEqual(payload["workflow_n_basins"], 1)
        self.assertTrue(payload["collapsed_all_basis_sites_into_one_basin"])
        self.assertEqual(payload["interpretation"], "suspicious_hollow_collapse")
        self.assertTrue(payload["physics_flags"]["coordinated_to_three_or_more_surface_atoms"])


if __name__ == "__main__":
    unittest.main()
