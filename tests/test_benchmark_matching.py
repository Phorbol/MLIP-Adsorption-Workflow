import unittest

from adsorption_ensemble.benchmark import select_unique_reference_matches


class TestBenchmarkMatching(unittest.TestCase):
    def test_select_unique_reference_matches_prevents_many_to_one_overcount(self):
        matches = select_unique_reference_matches(
            [
                {"manual_index": 0, "ours_index": 0, "signature_match": False, "rmsd": 0.10},
                {"manual_index": 1, "ours_index": 0, "signature_match": False, "rmsd": 0.11},
            ],
            n_manual=2,
            n_ours=1,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(len({int(row["ours_index"]) for row in matches}), 1)

    def test_select_unique_reference_matches_finds_two_distinct_pairs_when_available(self):
        matches = select_unique_reference_matches(
            [
                {"manual_index": 0, "ours_index": 0, "signature_match": False, "rmsd": 0.20},
                {"manual_index": 0, "ours_index": 1, "signature_match": False, "rmsd": 0.05},
                {"manual_index": 1, "ours_index": 0, "signature_match": False, "rmsd": 0.06},
            ],
            n_manual=2,
            n_ours=2,
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual({int(row["manual_index"]) for row in matches}, {0, 1})
        self.assertEqual({int(row["ours_index"]) for row in matches}, {0, 1})


if __name__ == "__main__":
    unittest.main()
