import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.audit_final_basin_identity import analyze_validation_payload, classify_extra_basin, write_audit_report


class TestFinalBasinIdentityAudit(unittest.TestCase):
    def test_classify_extra_basin_detects_nonmanual_anchor(self):
        out = classify_extra_basin(
            basin={
                "binding_adsorbate_indices": [1],
                "member_site_labels": ["ontop"],
            },
            manual_anchor_indices=[0],
            manual_site_names=["ontop", "bridge"],
        )
        self.assertEqual(out["primary_reason"], "binding_atom_outside_manual_anchor_set")

    def test_classify_extra_basin_falls_back_to_binding_pairs_when_binding_indices_missing(self):
        out = classify_extra_basin(
            basin={
                "binding_pairs": [[1, 10]],
                "member_site_labels": ["ontop"],
            },
            manual_anchor_indices=[0],
            manual_site_names=["ontop", "bridge"],
        )
        self.assertEqual(out["primary_reason"], "binding_atom_outside_manual_anchor_set")

    def test_analyze_validation_payload_counts_extra_cases(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            ours_work = root / "ours"
            manual_work = root / "ase_manual"
            ours_work.mkdir(parents=True, exist_ok=True)
            manual_work.mkdir(parents=True, exist_ok=True)
            (ours_work / "basins.json").write_text(
                json.dumps(
                    {
                        "basins": [
                            {
                                "basin_id": 0,
                                "energy_ev": -1.0,
                                "signature": "sig0",
                                "binding_adsorbate_indices": [0],
                                "binding_adsorbate_symbols": ["C"],
                                "binding_pairs": [[0, 10]],
                                "member_site_labels": ["ontop"],
                            },
                            {
                                "basin_id": 1,
                                "energy_ev": -0.5,
                                "signature": "sig1",
                                "binding_adsorbate_indices": [1],
                                "binding_adsorbate_symbols": ["O"],
                                "binding_pairs": [[1, 10]],
                                "member_site_labels": ["ontop"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (manual_work / "manual_basin_summary.json").write_text(
                json.dumps(
                    {
                        "manual_sites": [
                            {"site_name": "ontop", "mol_index": 0},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            payload = {
                "out_root": root.as_posix(),
                "rows": [
                    {
                        "case": "demo",
                        "ours": {"work_dir": ours_work.as_posix(), "n_basins": 2},
                        "ase_manual": {"work_dir": manual_work.as_posix(), "n_basins": 1},
                        "overlap": {
                            "matched_manual_basins": 1,
                            "n_manual_basins": 1,
                            "manual_recall_by_ours": 1.0,
                            "matches": [{"ours_index": 0}],
                        },
                    }
                ],
            }
            audit = analyze_validation_payload(payload)
            self.assertEqual(audit["n_cases"], 1)
            self.assertEqual(audit["n_extra_cases"], 1)
            self.assertEqual(audit["n_anchor_free_extra_binding_atom_basins"], 1)
            case = audit["case_audits"][0]
            self.assertEqual(case["n_extra_ours_basins"], 1)
            self.assertEqual(case["extra_ours_basins"][0]["classification"]["primary_reason"], "binding_atom_outside_manual_anchor_set")

            json_path, md_path = write_audit_report(audit, root / "report")
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
