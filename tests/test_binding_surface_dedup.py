import unittest

import numpy as np
from ase import Atoms
from ase.build import fcc111

from adsorption_ensemble.basin.dedup import build_binding_pairs, cluster_by_binding_pattern_and_surface_distance


def _linear_co() -> Atoms:
    return Atoms(
        "CO",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.15],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[True, True, False],
    )


def _top_layer_indices(slab: Atoms) -> list[int]:
    z = np.asarray(slab.get_positions(), dtype=float)[:, 2]
    zmax = float(np.max(z))
    return [int(i) for i, zi in enumerate(z.tolist()) if abs(float(zi) - zmax) <= 1.0e-6]


def _make_adsorbed_frame(slab: Atoms, site_xy: np.ndarray, carbon_height: float) -> Atoms:
    ads = _linear_co()
    shift = np.asarray([float(site_xy[0]), float(site_xy[1]), float(np.max(slab.get_positions()[:, 2]) + carbon_height)], dtype=float)
    ads.positions = np.asarray(ads.get_positions(), dtype=float) + shift[None, :]
    return slab.copy() + ads


class TestBindingSurfaceDedup(unittest.TestCase):
    def test_binding_surface_distance_merges_equivalent_top_sites_but_keeps_hollow_distinct(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=12.0)
        top_ids = _top_layer_indices(slab)
        self.assertGreaterEqual(len(top_ids), 3)
        top_pos = np.asarray(slab.get_positions(), dtype=float)[np.asarray(top_ids, dtype=int)]
        ref = top_pos[0]
        nn_order = np.argsort(np.linalg.norm(top_pos - ref[None, :], axis=1))
        top_a = top_pos[int(nn_order[0])]
        top_b = top_pos[int(nn_order[1])]
        top_c = top_pos[int(nn_order[2])]
        hollow_xy = np.mean(np.vstack([top_a[:2], top_b[:2], top_c[:2]]), axis=0)

        frame_top_1 = _make_adsorbed_frame(slab=slab, site_xy=top_a[:2], carbon_height=2.00)
        frame_top_2 = _make_adsorbed_frame(slab=slab, site_xy=top_b[:2], carbon_height=2.00)
        frame_hollow = _make_adsorbed_frame(slab=slab, site_xy=hollow_xy, carbon_height=1.30)
        frames = [frame_top_1, frame_top_2, frame_hollow]
        slab_n = len(slab)

        top_pairs = build_binding_pairs(frame_top_1, slab_n=slab_n, binding_tau=1.15)
        hollow_pairs = build_binding_pairs(frame_hollow, slab_n=slab_n, binding_tau=1.15)
        self.assertEqual(len({i for i, _ in top_pairs}), 1)
        self.assertEqual(len(top_pairs), 1)
        self.assertEqual(len({i for i, _ in hollow_pairs}), 1)
        self.assertGreaterEqual(len(hollow_pairs), 3)

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=frames,
            energies=np.zeros(len(frames), dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="binding_only",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="hierarchical",
        )

        member_sets = sorted(sorted(int(i) for i in basin["member_candidate_ids"]) for basin in basins)
        self.assertEqual(len(basins), 2)
        self.assertEqual(member_sets, [[0, 1], [2]])
        self.assertEqual(meta["surface_nearest_k"], 8)


if __name__ == "__main__":
    unittest.main()
