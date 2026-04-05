import unittest

from tools.audit_miller_monodentate_validation import build_audit, classify_case, reference_source_of_row


class TestMillerMonodentateValidationAudit(unittest.TestCase):
    def test_reference_source_defaults_to_ase_for_legacy_rows_with_input_sites(self):
        self.assertEqual(reference_source_of_row({"ase_manual": {"n_input_sites": 4}}), "ase_adsorbate_info")
        self.assertEqual(reference_source_of_row({"ase_manual": {"n_input_sites": 0}}), "")

    def test_classify_case_categories(self):
        self.assertEqual(
            classify_case({"ase_manual": {"n_input_sites": 0, "n_basins": 0}, "ours": {"n_basins": 1}, "overlap": {}}),
            "manual_reference_missing",
        )
        self.assertEqual(
            classify_case({"ase_manual": {"n_input_sites": 4, "n_basins": 0}, "ours": {"n_basins": 0}, "overlap": {}}),
            "both_zero",
        )
        self.assertEqual(
            classify_case({"ase_manual": {"n_input_sites": 4, "n_basins": 0}, "ours": {"n_basins": 1}, "overlap": {}}),
            "manual_zero_ours_positive",
        )
        self.assertEqual(
            classify_case({"ase_manual": {"n_input_sites": 4, "n_basins": 1}, "ours": {"n_basins": 0}, "overlap": {}}),
            "manual_positive_ours_zero",
        )
        self.assertEqual(
            classify_case({"ase_manual": {"n_input_sites": 4, "n_basins": 1}, "ours": {"n_basins": 1}, "overlap": {"manual_recall_by_ours": 1.0}}),
            "full_recall",
        )
        self.assertEqual(
            classify_case({"ase_manual": {"n_input_sites": 4, "n_basins": 2}, "ours": {"n_basins": 1}, "overlap": {"manual_recall_by_ours": 0.5}}),
            "partial_recall",
        )

    def test_build_audit_aggregates_status_counts(self):
        payload = {
            "out_root": "artifacts/example",
            "rows": [
                {
                    "case": "A",
                    "slab": "S1",
                    "adsorbate": "CO",
                    "ours": {"n_basins": 1, "n_basis_primitives": 4, "n_pose_frames": 16, "n_pose_frames_selected_for_basin": 16},
                    "ase_manual": {"reference_source": "ase_adsorbate_info", "n_input_sites": 4, "n_basins": 1, "rejected_reason_counts": {}},
                    "overlap": {"manual_recall_by_ours": 1.0},
                },
                {
                    "case": "B",
                    "slab": "S2",
                    "adsorbate": "H2O",
                    "ours": {"n_basins": 0, "n_basis_primitives": 3, "n_pose_frames": 12, "n_pose_frames_selected_for_basin": 12},
                    "ase_manual": {"reference_source": "primitive_basis_fallback", "n_input_sites": 3, "n_basins": 0, "rejected_reason_counts": {"desorption": 3}},
                    "overlap": {"manual_recall_by_ours": None, "manual_reference_state": "empty_agreement"},
                },
                {
                    "case": "C",
                    "slab": "S3",
                    "adsorbate": "NH3",
                    "ours": {"n_basins": 1, "n_basis_primitives": 8, "n_pose_frames": 20, "n_pose_frames_selected_for_basin": 20},
                    "ase_manual": {"n_input_sites": 0, "n_basins": 0, "rejected_reason_counts": {}},
                    "overlap": {"manual_recall_by_ours": 0.0},
                    "sentinel_audit": {"interpretation": "suspicious_hollow_collapse", "final_binding_environment": {"coordination": 3}},
                },
            ],
        }
        nullzone_payload = {
            "case_diagnosis": [
                {"case": "B", "all_modes_zero_basins": True, "aggregate_rejected_reason_counts": {"desorption": 3}}
            ]
        }
        audit = build_audit(payload, nullzone_payload=nullzone_payload)
        self.assertEqual(audit["n_cases"], 3)
        self.assertEqual(audit["status_counts"]["full_recall"], 1)
        self.assertEqual(audit["status_counts"]["robust_null_zone"], 1)
        self.assertEqual(audit["status_counts"]["manual_reference_missing"], 1)
        self.assertEqual(audit["reference_source_counts"]["ase_adsorbate_info"], 1)
        self.assertEqual(audit["reference_source_counts"]["primitive_basis_fallback"], 1)
        self.assertEqual(audit["recall_defined_cases"], 1)
        self.assertAlmostEqual(audit["manual_defined_mean_recall"], 1.0)
        self.assertEqual(len(audit["sentinel_cases"]), 1)
        self.assertEqual(audit["manual_positive_full_recall_cases"], 1)
        self.assertEqual(len(audit["robust_null_zone_cases"]), 1)


if __name__ == "__main__":
    unittest.main()
