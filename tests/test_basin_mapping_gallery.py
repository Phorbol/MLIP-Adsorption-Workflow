import unittest

import numpy as np
from ase import Atoms
from ase.build import fcc111

from adsorption_ensemble.basin.dedup import (
    binding_signature,
    build_binding_pairs,
    cluster_by_binding_pattern_and_surface_distance,
)


def _co_adsorbate() -> Atoms:
    return Atoms(
        "CO",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.15],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[True, True, False],
    )


def _oco_adsorbate() -> Atoms:
    return Atoms(
        "OCO",
        positions=[
            [-1.18, 0.0, -0.45],
            [0.0, 0.0, 0.0],
            [1.18, 0.0, -0.45],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _heterogeneous_support() -> Atoms:
    return Atoms(
        symbols=["Pt", "Cu", "Pt", "Pt"],
        positions=[
            [-1.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.7, 0.0],
            [0.0, -1.7, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _bidentate_support() -> Atoms:
    return Atoms(
        symbols=["Pt", "Pt", "Pt"],
        positions=[
            [-1.35, 0.0, 0.0],
            [1.35, 0.0, 0.0],
            [0.0, 2.2, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _co_top_frame(slab: Atoms, atom_index: int, *, carbon_height: float = 1.95) -> Atoms:
    ads = _co_adsorbate().copy()
    ads.positions = np.asarray(ads.positions, dtype=float) + np.asarray(slab.positions[int(atom_index)], dtype=float) + np.asarray(
        [0.0, 0.0, float(carbon_height)],
        dtype=float,
    )
    return slab.copy() + ads


def _oco_bidentate_frame(
    *,
    shift: np.ndarray | list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    monodentate: bool = False,
) -> Atoms:
    slab = _bidentate_support()
    ads = _oco_adsorbate().copy()
    ads.positions = np.asarray(ads.positions, dtype=float) + np.asarray([0.0, 0.0, 2.05], dtype=float) + np.asarray(shift, dtype=float)
    if bool(monodentate):
        ads.positions[2, 2] += 1.8
    return slab + ads


def _pt111_top_and_bridge_frames() -> tuple[Atoms, Atoms, Atoms, int]:
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=12.0)
    z = np.asarray(slab.positions[:, 2], dtype=float)
    zmax = float(np.max(z))
    top_ids = [int(i) for i, zi in enumerate(z.tolist()) if abs(float(zi) - zmax) < 1.0e-8]
    top_pos = np.asarray(slab.positions, dtype=float)[np.asarray(top_ids, dtype=int)]
    ref = top_pos[0]
    order = np.argsort(np.linalg.norm(top_pos - ref[None, :], axis=1))
    top_a = top_pos[int(order[0])]
    top_b = top_pos[int(order[1])]
    bridge_xy = 0.5 * (top_a[:2] + top_b[:2])

    def _frame_xy(xy: np.ndarray, carbon_height: float) -> Atoms:
        ads = _co_adsorbate().copy()
        ads.positions = np.asarray(ads.positions, dtype=float) + np.asarray([float(xy[0]), float(xy[1]), zmax + float(carbon_height)], dtype=float)
        return slab.copy() + ads

    return _frame_xy(top_a[:2], 2.0), _frame_xy(top_b[:2], 2.0), _frame_xy(bridge_xy, 1.55), len(slab)


class TestBasinMappingGallery(unittest.TestCase):
    def test_binding_surface_distance_merges_pt111_equivalent_top_and_splits_bridge(self):
        frame_top_a, frame_top_b, frame_bridge, slab_n = _pt111_top_and_bridge_frames()

        self.assertEqual(len(build_binding_pairs(frame_top_a, slab_n=slab_n, binding_tau=1.15)), 1)
        self.assertEqual(len(build_binding_pairs(frame_top_b, slab_n=slab_n, binding_tau=1.15)), 1)
        self.assertEqual(len(build_binding_pairs(frame_bridge, slab_n=slab_n, binding_tau=1.15)), 2)

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_top_a, frame_top_b, frame_bridge],
            energies=np.asarray([0.0, 0.01, 0.02], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="binding_only",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="hierarchical",
        )

        self.assertEqual(len(basins), 2)
        self.assertEqual(sorted(sorted(int(i) for i in basin["member_candidate_ids"]) for basin in basins), [[0, 1], [2]])
        self.assertEqual(meta["surface_atom_mode"], "binding_only")

    def test_binding_surface_distance_currently_collapses_chemically_distinct_equivalent_top_sites(self):
        slab = _heterogeneous_support()
        frame_pt_top = _co_top_frame(slab, 0)
        frame_cu_top = _co_top_frame(slab, 1)
        slab_n = len(slab)

        pairs_pt = build_binding_pairs(frame_pt_top, slab_n=slab_n, binding_tau=1.15)
        pairs_cu = build_binding_pairs(frame_cu_top, slab_n=slab_n, binding_tau=1.15)
        self.assertEqual(pairs_pt, [(0, 0)])
        self.assertEqual(pairs_cu, [(0, 1)])
        self.assertNotEqual(
            binding_signature(pairs_pt, frame=frame_pt_top, slab_n=slab_n, mode="canonical"),
            binding_signature(pairs_cu, frame=frame_cu_top, slab_n=slab_n, mode="canonical"),
        )

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_pt_top, frame_cu_top],
            energies=np.asarray([0.0, 0.02], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="binding_only",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="greedy",
        )

        # This documents a current robustness gap: the distance-only local
        # surface descriptor ignores slab atom species, so geometrically
        # equivalent Pt-top and Cu-top states collapse under the default metric.
        self.assertEqual(len(basins), 1)
        self.assertEqual(basins[0]["member_candidate_ids"], [0, 1])
        self.assertEqual(meta["surface_atom_mode"], "binding_only")

    def test_binding_surface_distance_merges_micro_perturbed_bidentate_oco_and_splits_monodentate(self):
        frame_bidentate_a = _oco_bidentate_frame()
        frame_bidentate_b = _oco_bidentate_frame(shift=[0.03, 0.0, 0.02])
        frame_monodentate = _oco_bidentate_frame(monodentate=True)
        slab_n = len(_bidentate_support())

        self.assertEqual(build_binding_pairs(frame_bidentate_a, slab_n=slab_n, binding_tau=1.15), [(0, 0), (2, 1)])
        self.assertEqual(build_binding_pairs(frame_bidentate_b, slab_n=slab_n, binding_tau=1.15), [(0, 0), (2, 1)])
        self.assertEqual(build_binding_pairs(frame_monodentate, slab_n=slab_n, binding_tau=1.15), [(0, 0)])

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_bidentate_a, frame_bidentate_b, frame_monodentate],
            energies=np.asarray([0.0, 0.01, 0.03], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="binding_only",
            surface_relative=False,
            surface_rmsd_gate=0.35,
            cluster_method="greedy",
        )

        self.assertEqual(len(basins), 2)
        self.assertEqual(sorted(sorted(int(i) for i in basin["member_candidate_ids"]) for basin in basins), [[0, 1], [2]])
        self.assertEqual(meta["surface_atom_mode"], "binding_only")


if __name__ == "__main__":
    unittest.main()
