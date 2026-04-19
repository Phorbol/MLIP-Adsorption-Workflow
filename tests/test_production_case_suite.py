import unittest
from pathlib import Path

from tools.run_production_case_suite import (
    build_pt211_ag4_slab,
    default_case_specs,
    load_cnb_isomer_adsorbates,
)


class TestProductionCaseSuiteHelpers(unittest.TestCase):
    def test_load_cnb_isomer_adsorbates_from_examples(self):
        adsorbates = load_cnb_isomer_adsorbates(Path("examples"))
        self.assertEqual(set(adsorbates), {"oCNB", "mCNB", "pCNB"})
        for name, atoms in adsorbates.items():
            with self.subTest(name=name):
                self.assertEqual(atoms.get_chemical_formula(), "C6H4ClNO2")
                self.assertEqual(len(atoms), 14)

    def test_build_pt211_ag4_slab_adds_four_silver_atoms(self):
        slab = build_pt211_ag4_slab()
        symbols = slab.get_chemical_symbols()
        self.assertEqual(symbols.count("Ag"), 4)
        self.assertGreater(symbols.count("Pt"), 0)
        self.assertEqual(tuple(bool(x) for x in slab.get_pbc()), (True, True, False))

    def test_default_case_specs_cover_baseline_cnb_and_heterogeneous_cases(self):
        ids = {spec.case_id for spec in default_case_specs()}
        self.assertIn("fcc111__NH3", ids)
        self.assertIn("Pt_fcc211__CO", ids)
        self.assertIn("Pt_fcc211__C2H2", ids)
        self.assertIn("Pt_fcc211__oCNB", ids)
        self.assertIn("Pt_fcc211__mCNB", ids)
        self.assertIn("Pt_fcc211__pCNB", ids)
        self.assertIn("Pt211Ag4__CO", ids)
        self.assertIn("Pt211Ag4__C6H6", ids)


if __name__ == "__main__":
    unittest.main()
