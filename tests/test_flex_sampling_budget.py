import unittest

from ase.build import molecule

from adsorption_ensemble.workflows import plan_flex_sampling_budget
from tests.chemistry_cases import make_dipeptide_like, make_para_nitrobenzoic_acid_like


class TestFlexSamplingBudget(unittest.TestCase):
    def test_small_rigid_adsorbate_prefers_no_conformer(self):
        b = plan_flex_sampling_budget(molecule("CO"), n_surface_atoms=16, n_site_primitives=24)
        self.assertFalse(b.run_conformer_search)
        self.assertGreaterEqual(b.score, 0.0)

    def test_large_flexible_adsorbate_enables_conformer(self):
        b = plan_flex_sampling_budget(make_dipeptide_like(), n_surface_atoms=32, n_site_primitives=64)
        self.assertTrue(b.run_conformer_search)
        self.assertGreater(b.md_time_ps, 0.0)
        self.assertGreaterEqual(b.md_runs, 2)
        self.assertGreaterEqual(b.preselect_k, 64)

    def test_functionalized_aromatic_tends_to_enable_conformer(self):
        b = plan_flex_sampling_budget(make_para_nitrobenzoic_acid_like(), n_surface_atoms=24, n_site_primitives=48)
        self.assertGreaterEqual(b.score, 2.0)
        self.assertIn("rotatable_proxy", b.rationale)


if __name__ == "__main__":
    unittest.main()

