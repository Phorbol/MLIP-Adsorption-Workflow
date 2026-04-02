import unittest

import numpy as np
from ase import Atoms
from ase.build import fcc111

from adsorption_ensemble.basin.dedup import (
    binding_signature,
    build_binding_pairs,
    cluster_by_signature_and_rmsd,
    kabsch_rmsd,
    symmetry_aware_kabsch_rmsd,
)


class TestBasinSignatureModes(unittest.TestCase):
    def test_canonical_signature_merges_equivalent_top_sites(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 2)

        frame_a = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.1])])
        frame_b = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.1])])
        pairs_a = build_binding_pairs(frame_a, slab_n=len(slab), binding_tau=1.15)
        pairs_b = build_binding_pairs(frame_b, slab_n=len(slab), binding_tau=1.15)

        self.assertNotEqual(binding_signature(pairs_a), binding_signature(pairs_b))
        self.assertEqual(
            binding_signature(pairs_a, frame=frame_a, slab_n=len(slab), mode="canonical"),
            binding_signature(pairs_b, frame=frame_b, slab_n=len(slab), mode="canonical"),
        )

    def test_pure_rmsd_can_merge_across_absolute_signatures(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        frame_a = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.1])])
        frame_b = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.1])])
        basins = cluster_by_signature_and_rmsd(
            frames=[frame_a, frame_b],
            energies=np.asarray([0.0, 0.01], dtype=float),
            slab_n=len(slab),
            binding_tau=1.15,
            rmsd_threshold=1e-8,
            signature_mode="absolute",
            use_signature_grouping=False,
        )
        self.assertEqual(len(basins), 1)

    def test_provenance_signature_uses_site_label(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        frame_a = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.1])])
        frame_b = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.1])])
        frame_a.info["site_label"] = "ontop"
        frame_b.info["site_label"] = "ontop"
        pairs_a = build_binding_pairs(frame_a, slab_n=len(slab), binding_tau=1.15)
        pairs_b = build_binding_pairs(frame_b, slab_n=len(slab), binding_tau=1.15)
        self.assertEqual(
            binding_signature(pairs_a, frame=frame_a, slab_n=len(slab), mode="provenance"),
            binding_signature(pairs_b, frame=frame_b, slab_n=len(slab), mode="provenance"),
        )

    def test_none_signature_mode_collapses_grouping(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        frame_a = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.1])])
        frame_b = slab.copy() + Atoms("H", positions=[slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.1])])
        pairs_a = build_binding_pairs(frame_a, slab_n=len(slab), binding_tau=1.15)
        pairs_b = build_binding_pairs(frame_b, slab_n=len(slab), binding_tau=1.15)
        self.assertEqual(binding_signature(pairs_a, mode="none"), "all")
        self.assertEqual(binding_signature(pairs_b, mode="none"), "all")

    def test_symmetry_aware_rmsd_handles_permuted_equivalent_atoms(self):
        ads = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],
                [0.94, 0.0, -0.30],
                [-0.47, 0.81, -0.30],
                [-0.47, -0.81, -0.30],
            ],
        )
        permuted = ads[[0, 2, 3, 1]]
        plain = kabsch_rmsd(ads.positions, permuted.positions)
        symm = symmetry_aware_kabsch_rmsd(ads.positions, permuted.positions, ads)
        self.assertLessEqual(symm, plain + 1e-12)
        self.assertLess(symm, 1e-8)

    def test_cluster_rmsd_merges_symmetric_nh3_permutations(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],
                [0.94, 0.0, -0.30],
                [-0.47, 0.81, -0.30],
                [-0.47, -0.81, -0.30],
            ],
        )
        frame_a = slab.copy() + ads
        frame_a.info["site_label"] = "bridge"
        frame_b = slab.copy() + ads[[0, 2, 3, 1]]
        frame_b.info["site_label"] = "bridge"
        basins = cluster_by_signature_and_rmsd(
            frames=[frame_a, frame_b],
            energies=np.asarray([0.0, 0.01], dtype=float),
            slab_n=len(slab),
            binding_tau=0.0,
            rmsd_threshold=0.05,
            signature_mode="provenance",
            use_signature_grouping=True,
        )
        self.assertEqual(len(basins), 1)


if __name__ == "__main__":
    unittest.main()
