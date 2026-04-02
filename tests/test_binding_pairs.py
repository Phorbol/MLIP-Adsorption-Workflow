import unittest

from ase import Atoms

from adsorption_ensemble.basin.anomaly import classify_anomaly
from adsorption_ensemble.basin.dedup import build_binding_pairs


class TestBindingPairs(unittest.TestCase):
    def test_binding_pairs_ignore_hydrogen_contacts_when_heavy_atom_present(self):
        frame = Atoms(
            "PtOH",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.60],
                [0.0, 0.0, 1.60],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=[False, False, False],
        )
        pairs = build_binding_pairs(frame, slab_n=1, binding_tau=1.15)
        self.assertEqual(pairs, [])

    def test_binding_pairs_keep_hydrogen_for_h_only_adsorbate(self):
        frame = Atoms(
            "PtH",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.60],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=[False, False, False],
        )
        pairs = build_binding_pairs(frame, slab_n=1, binding_tau=1.15)
        self.assertEqual(pairs, [(0, 0)])

    def test_anomaly_treats_hydrogen_only_contact_from_heavy_adsorbate_as_desorption(self):
        slab = Atoms(
            "Pt",
            positions=[[0.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=[False, False, False],
        )
        adsorbate = Atoms(
            "OH",
            positions=[
                [0.0, 0.0, 2.60],
                [0.0, 0.0, 1.60],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=[False, False, False],
        )
        relaxed = slab + adsorbate
        reason, metrics = classify_anomaly(
            relaxed=relaxed,
            slab_ref=slab,
            adsorbate_ref=adsorbate,
            slab_n=1,
            normal_axis=2,
            binding_tau=1.15,
            desorption_min_bonds=1,
            surface_reconstruction_max_disp=10.0,
            dissociation_allow_bond_change=False,
            burial_margin=10.0,
        )
        self.assertEqual(reason, "desorption")
        self.assertEqual(metrics["binding_pair_n"], 0)


if __name__ == "__main__":
    unittest.main()
