import unittest
from collections import Counter

import numpy as np
from ase.build import fcc100, fcc111

from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import SurfacePreprocessor


class TestSiteEmbeddingPhysics(unittest.TestCase):
    def _embed_counts(self, slab):
        ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
        raw = PrimitiveBuilder().build(slab, ctx)
        z = slab.get_atomic_numbers().astype(float)
        feats = (z / (np.max(z) + 1e-12)).reshape(-1, 1)
        res = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.22)).fit_transform(
            slab=slab,
            primitives=list(raw),
            atom_features=feats,
        )
        raw_by_kind = Counter([p.kind for p in res.primitives])
        basis_by_kind = Counter([p.kind for p in res.basis_primitives])
        return raw_by_kind, basis_by_kind

    def test_fcc111_distinguishes_fcc_and_hcp_hollow(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=10.0)
        raw, basis = self._embed_counts(slab)
        self.assertGreater(raw.get("3c", 0), 0)
        # fcc(111) should keep both fcc and hcp hollow families as inequivalent basins/sites.
        self.assertEqual(basis.get("3c", 0), 2)

    def test_fcc100_keeps_single_hollow_family(self):
        slab = fcc100("Pt", size=(4, 4, 4), vacuum=10.0)
        raw, basis = self._embed_counts(slab)
        self.assertGreater(raw.get("4c", 0), 0)
        # fcc(100) square hollow should remain one inequivalent family.
        self.assertEqual(basis.get("4c", 0), 1)


if __name__ == "__main__":
    unittest.main()

