import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from ase.build import fcc111, molecule
from ase.io import write

from adsorption_ensemble.basin.types import Basin
from adsorption_ensemble.node import NodeConfig, basin_to_node
from tools.audit_symmetry_canonical_nodes import analyze_case, write_report


def _pack_group(source_rows: list[dict], merged_basin_id: int) -> dict:
    energies = [float(row["energy_ev"]) for row in source_rows]
    observed_node_ids = sorted({str(row["node_id"]) for row in source_rows})
    legacy_node_ids = sorted({str(row["node_id_legacy"]) for row in source_rows})
    surface_geometry_node_ids = sorted({str(row["node_id_surface_geometry"]) for row in source_rows})
    surface_env_only_keys = sorted({str(row["surface_env_only_key"]) for row in source_rows})
    surface_geometry_keys = sorted({str(row["surface_geometry_key"]) for row in source_rows})
    return {
        "merged_basin_id": int(merged_basin_id),
        "source_basin_ids": [int(row["basin_id"]) for row in source_rows],
        "source_node_ids": [str(row["node_id"]) for row in source_rows],
        "source_observed_node_ids": observed_node_ids,
        "source_legacy_node_ids": legacy_node_ids,
        "source_surface_geometry_node_ids": surface_geometry_node_ids,
        "source_signatures": sorted({str(row["signature"]) for row in source_rows}),
        "surface_env_only_keys": surface_env_only_keys,
        "surface_geometry_keys": surface_geometry_keys,
        "n_unique_observed_node_ids": int(len(observed_node_ids)),
        "n_unique_legacy_node_ids": int(len(legacy_node_ids)),
        "n_unique_surface_geometry_node_ids": int(len(surface_geometry_node_ids)),
        "n_unique_surface_env_only_keys": int(len(surface_env_only_keys)),
        "n_unique_surface_geometry_keys": int(len(surface_geometry_keys)),
        "energy_min_ev": float(min(energies)),
        "energy_max_ev": float(max(energies)),
        "energy_span_ev": float(max(energies) - min(energies)),
        "relative_energies_ev": [float(row["relative_energy_ev"]) for row in source_rows],
    }


class TestSymmetryCanonicalNodeAudit(unittest.TestCase):
    def _write_case(self, case_dir: Path) -> None:
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 2)
        ads = molecule("CO")

        frame_top_a = slab.copy() + ads.copy()
        frame_top_a.positions[len(slab) :] += slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.8])
        frame_top_b = slab.copy() + ads.copy()
        frame_top_b.positions[len(slab) :] += slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.8])
        frame_bridge = slab.copy() + ads.copy()
        frame_bridge.positions[len(slab) :] += 0.5 * (slab.positions[int(top_ids[0])] + slab.positions[int(top_ids[1])]) + np.array(
            [0.0, 0.0, 1.8]
        )

        basins = [
            Basin(
                basin_id=0,
                atoms=frame_top_a,
                energy_ev=0.0,
                member_candidate_ids=[0],
                binding_pairs=[(0, int(top_ids[0]))],
                denticity=1,
                signature="sig_top",
            ),
            Basin(
                basin_id=1,
                atoms=frame_top_b,
                energy_ev=0.01,
                member_candidate_ids=[1],
                binding_pairs=[(0, int(top_ids[1]))],
                denticity=1,
                signature="sig_top",
            ),
            Basin(
                basin_id=2,
                atoms=frame_bridge,
                energy_ev=0.02,
                member_candidate_ids=[2],
                binding_pairs=[(0, int(top_ids[0])), (0, int(top_ids[1]))],
                denticity=1,
                signature="sig_bridge",
            ),
        ]
        energy_min = min(float(b.energy_ev) for b in basins)
        nodes = [basin_to_node(b, slab_n=len(slab), cfg=NodeConfig(), energy_min_ev=energy_min, surface_reference=slab) for b in basins]

        case_dir.mkdir(parents=True, exist_ok=True)
        write((case_dir / "basins.extxyz").as_posix(), [b.atoms for b in basins])
        write((case_dir / "slab_input.xyz").as_posix(), slab)
        (case_dir / "nodes.json").write_text(
            json.dumps(
                [
                    {
                        "node_id": str(n.node_id),
                        "node_id_legacy": str(n.node_id_legacy),
                        "basin_id": int(n.basin_id),
                        "canonical_order": [int(x) for x in n.canonical_order],
                        "atomic_numbers": [int(x) for x in n.atomic_numbers],
                        "internal_bonds": [(int(i), int(j)) for i, j in n.internal_bonds],
                        "binding_pairs": [(int(i), int(j)) for i, j in n.binding_pairs],
                        "surface_env_key": (None if n.surface_env_key is None else str(n.surface_env_key)),
                        "surface_geometry_key": (None if n.surface_geometry_key is None else str(n.surface_geometry_key)),
                        "denticity": int(n.denticity),
                        "relative_energy_ev": (None if n.relative_energy_ev is None else float(n.relative_energy_ev)),
                        "provenance": dict(n.provenance),
                    }
                    for n in nodes
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (case_dir / "basin_dictionary.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "energy_min_ev": float(energy_min),
                        "final_basin_merge": {
                            "model_path": "/tmp/fake-mace.model",
                            "dtype": "float32",
                            "max_edges_per_batch": 1000,
                            "layers_to_keep": -1,
                            "enable_cueq": False,
                        },
                    },
                    "false_split_suspect_signatures": ["sig_top"],
                    "basins": [
                        {
                            "basin_id": int(b.basin_id),
                            "energy_ev": float(b.energy_ev),
                            "denticity": int(b.denticity),
                            "signature": str(b.signature),
                            "member_candidate_ids": [int(x) for x in b.member_candidate_ids],
                            "binding_pairs": [(int(i), int(j)) for i, j in b.binding_pairs],
                        }
                        for b in basins
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_analyze_case_emits_formal_node_inflation_audit_and_detects_overmerge(self):
        with TemporaryDirectory() as td:
            case_dir = Path(td) / "Pd111_2x2__CO"
            self._write_case(case_dir)

            def _fake_threshold_merge_groups(*, basin_rows, use_signature_grouping, **kwargs):
                top_group = _pack_group([basin_rows[0], basin_rows[1]], merged_basin_id=0)
                bridge_group = _pack_group([basin_rows[2]], merged_basin_id=1)
                if use_signature_grouping:
                    payload = {
                        "0.05": {"n_output_basins": 2, "use_signature_grouping": True, "meta": {}, "groups": [top_group, bridge_group]},
                        "0.08": {"n_output_basins": 2, "use_signature_grouping": True, "meta": {}, "groups": [top_group, bridge_group]},
                    }
                else:
                    payload = {
                        "0.05": {"n_output_basins": 1, "use_signature_grouping": False, "meta": {}, "groups": [_pack_group(basin_rows, merged_basin_id=0)]},
                        "0.08": {"n_output_basins": 1, "use_signature_grouping": False, "meta": {}, "groups": [_pack_group(basin_rows, merged_basin_id=0)]},
                    }
                return {"descriptor_meta": {"provided_by_test": True}, "thresholds": payload}

            with mock.patch("tools.audit_symmetry_canonical_nodes._threshold_merge_groups", side_effect=_fake_threshold_merge_groups):
                case = analyze_case(case_dir=case_dir, thresholds=[0.05, 0.08], device="cpu")

            self.assertEqual(case["counts"]["current_node_ids"], 3)
            self.assertEqual(case["counts"]["surface_geometry_node_ids"], 2)
            self.assertEqual(case["counts"]["inflation_observed_vs_surface_geometry"], 1)
            self.assertEqual(case["counts"]["redundant_surface_geometry_split_groups"], 1)
            self.assertEqual(case["downstream_node_inflation_audit"]["grouped"]["0.05"]["delta_vs_surface_geometry_node_ids"], 0)
            self.assertEqual(case["downstream_node_inflation_audit"]["grouped"]["0.05"]["same_surface_geometry_resolutions"], 1)
            self.assertEqual(case["downstream_node_inflation_audit"]["grouped"]["0.05"]["cross_surface_geometry_merges"], 0)
            self.assertEqual(case["downstream_node_inflation_audit"]["ungrouped"]["0.05"]["delta_vs_surface_geometry_node_ids"], -1)
            self.assertEqual(case["downstream_node_inflation_audit"]["ungrouped"]["0.05"]["cross_surface_geometry_merges"], 1)
            self.assertEqual(case["threshold_merge_audit"]["grouped"]["thresholds"]["0.05"]["groups"][0]["n_unique_surface_geometry_keys"], 1)
            self.assertEqual(case["threshold_merge_audit"]["ungrouped"]["thresholds"]["0.05"]["groups"][0]["n_unique_surface_geometry_keys"], 2)

    def test_write_report_mentions_node_inflation_audit(self):
        report = {
            "run_dir": "runs/demo",
            "device": "cpu",
            "thresholds": [0.05],
            "cases": [
                {
                    "case": "demo_case",
                    "counts": {
                        "current_node_ids": 3,
                        "legacy_node_ids": 3,
                        "surface_geometry_node_ids": 2,
                        "inflation_observed_vs_surface_geometry": 1,
                        "inflation_legacy_vs_surface_geometry": 1,
                        "redundant_surface_geometry_split_groups": 1,
                        "redundant_surface_geometry_split_basins": 2,
                        "signatures": 2,
                        "surface_env_only_keys": 2,
                        "surface_geometry_keys": 2,
                        "threshold_0p05_grouped": 2,
                        "threshold_0p08_grouped": -1,
                        "threshold_0p05_ungrouped": 1,
                        "threshold_0p08_ungrouped": -1,
                    },
                    "n_basins_original": 3,
                    "false_split_suspect_signatures": ["sig_top"],
                    "node_inflation_audit": {
                        "redundant_surface_geometry_splits": [
                            {
                                "surface_geometry_key": "geom-top",
                                "basin_ids": [0, 1],
                                "observed_node_ids": ["node-a", "node-b"],
                                "energy_span_ev": 0.01,
                            }
                        ]
                    },
                    "surface_geometry_groups": [
                        {
                            "key": "geom-top",
                            "basin_ids": [0, 1],
                            "node_ids": ["node-a", "node-b"],
                            "signatures": ["sig_top"],
                            "energy_span_ev": 0.01,
                        }
                    ],
                    "downstream_node_inflation_audit": {
                        "grouped": {
                            "0.05": {
                                "delta_vs_surface_geometry_node_ids": 0,
                                "same_surface_geometry_resolutions": 1,
                                "cross_surface_geometry_merges": 0,
                            }
                        },
                        "ungrouped": {
                            "0.05": {
                                "delta_vs_surface_geometry_node_ids": -1,
                                "same_surface_geometry_resolutions": 0,
                                "cross_surface_geometry_merges": 1,
                            }
                        },
                    },
                    "threshold_merge_audit": {
                        "grouped": {
                            "thresholds": {
                                "0.05": {
                                    "n_output_basins": 2,
                                    "groups": [
                                        {
                                            "merged_basin_id": 0,
                                            "source_basin_ids": [0, 1],
                                            "source_observed_node_ids": ["node-a", "node-b"],
                                            "source_surface_geometry_node_ids": ["geom-node-top"],
                                            "surface_geometry_keys": ["geom-top"],
                                            "energy_span_ev": 0.01,
                                        }
                                    ],
                                }
                            }
                        },
                        "ungrouped": {
                            "thresholds": {
                                "0.05": {
                                    "n_output_basins": 1,
                                    "groups": [
                                        {
                                            "merged_basin_id": 0,
                                            "source_basin_ids": [0, 1, 2],
                                            "source_observed_node_ids": ["node-a", "node-b", "node-c"],
                                            "source_surface_geometry_node_ids": ["geom-node-top", "geom-node-bridge"],
                                            "surface_geometry_keys": ["geom-top", "geom-bridge"],
                                            "energy_span_ev": 0.02,
                                        }
                                    ],
                                }
                            }
                        },
                    },
                }
            ],
        }
        with TemporaryDirectory() as td:
            json_path, md_path = write_report(report, Path(td) / "audit")
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("### Node Inflation Audit", md_text)
            self.assertIn("delta_vs_surface_geometry", md_text)
            self.assertIn("surface_geometry_node_ids", md_text)


if __name__ == "__main__":
    unittest.main()
