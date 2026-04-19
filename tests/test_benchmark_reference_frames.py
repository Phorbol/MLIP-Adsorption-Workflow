import unittest

import numpy as np
from ase import Atoms
from ase.build import fcc111, fcc211, molecule

from adsorption_ensemble.benchmark import build_ase_reference_frames, manual_reference_height


class TestBenchmarkReferenceFrames(unittest.TestCase):
    def test_manual_reference_height_prefers_lower_hollow_height_for_atom(self):
        ads = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        self.assertLess(manual_reference_height(ads, "fcc"), manual_reference_height(ads, "ontop"))

    def test_build_ase_reference_frames_uses_heavy_donor_for_h2o_nh3_and_methanol(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        cases = {
            "H2O": "O",
            "NH3": "N",
            "CH3OH": "O",
        }
        for name, expected_symbol in cases.items():
            ads = molecule(name)
            frames, meta = build_ase_reference_frames(slab=slab, adsorbate=ads)
            self.assertEqual(len(frames), 4)
            self.assertEqual(len(meta), 4)
            self.assertEqual(str(meta[0]["binding_atom_symbol"]), expected_symbol)
            for frame in frames:
                slab_n = len(slab)
                ads_part = frame[slab_n:]
                binding_index = int(frame.info["binding_atom_index"])
                self.assertEqual(str(frame.info["binding_atom_symbol"]), expected_symbol)
                other_ids = [i for i in range(len(ads_part)) if i != binding_index]
                if other_ids:
                    self.assertLess(
                        float(ads_part.positions[binding_index, 2]),
                        float(np.mean(ads_part.positions[other_ids, 2])),
                    )

    def test_build_ase_reference_frames_keeps_carbon_down_for_co(self):
        slab = fcc111("Cu", size=(4, 4, 4), vacuum=12.0)
        ads = molecule("CO")
        if ads[0].symbol != "C":
            ads = ads[[1, 0]]
        frames, meta = build_ase_reference_frames(slab=slab, adsorbate=ads)
        self.assertEqual(len(frames), 4)
        self.assertEqual(str(meta[0]["binding_atom_symbol"]), "C")
        for frame in frames:
            ads_part = frame[len(slab):]
            binding_index = int(frame.info["binding_atom_index"])
            other_ids = [i for i in range(len(ads_part)) if i != binding_index]
            self.assertLess(
                float(ads_part.positions[binding_index, 2]),
                float(np.mean(ads_part.positions[other_ids, 2])),
            )

    def test_build_ase_reference_frames_falls_back_to_primitive_basis_for_fcc211(self):
        slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
        ads = molecule("NH3")
        frames, meta = build_ase_reference_frames(slab=slab, adsorbate=ads)
        self.assertGreaterEqual(len(frames), 4)
        self.assertEqual(len(frames), len(meta))
        self.assertTrue(all(str(row["reference_source"]) == "primitive_basis_fallback" for row in meta))
        self.assertTrue(all(str(frame.info["reference_source"]) == "primitive_basis_fallback" for frame in frames))
        self.assertTrue(all(str(row["site_kind"]) in {"1c", "2c", "3c", "4c"} for row in meta))
        unique_labels = {str(row["site_label"]) for row in meta}
        self.assertEqual(len(unique_labels), len(meta))


if __name__ == "__main__":
    unittest.main()
