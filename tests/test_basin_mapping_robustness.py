import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms

from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.basin.dedup import (
    build_binding_pairs,
    cluster_by_binding_pattern_and_surface_distance,
)


def _anisotropic_support() -> Atoms:
    return Atoms(
        symbols=["Pt", "Cu", "Pt", "Pt"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.0],
            [0.6, 1.7, 0.0],
            [-1.5, 0.4, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _threefold_support() -> Atoms:
    return Atoms(
        symbols=["Pt", "Pt", "Pt"],
        positions=[
            [1.2, 0.0, 0.0],
            [-0.6, 1.03923048, 0.0],
            [-0.6, -1.03923048, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _co_adsorbate() -> Atoms:
    return Atoms(
        "CO",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.15],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _nh3_adsorbate() -> Atoms:
    return Atoms(
        "NH3",
        positions=[
            [0.0, 0.0, 0.0],
            [0.94, 0.0, -0.34],
            [-0.47, 0.81, -0.34],
            [-0.47, -0.81, -0.34],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[False, False, False],
    )


def _rotz(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(theta_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _co_frame(*, shift: np.ndarray | list[float] | tuple[float, float, float]) -> Atoms:
    slab = _anisotropic_support()
    ads = _co_adsorbate().copy()
    ads.positions = np.asarray(ads.positions, dtype=float) + np.asarray([0.0, 0.0, 1.95], dtype=float) + np.asarray(shift, dtype=float)
    return slab + ads


def _nh3_frame(theta_deg: float) -> Atoms:
    slab = _anisotropic_support()
    ads = _nh3_adsorbate().copy()
    ads.positions = np.asarray(ads.positions, dtype=float) @ _rotz(theta_deg).T
    ads.positions = np.asarray(ads.positions, dtype=float) + np.asarray([0.0, 0.0, 1.95], dtype=float)
    return slab + ads


def _nh3_frame_on_threefold_support(theta_deg: float) -> Atoms:
    slab = _threefold_support()
    ads = _nh3_adsorbate().copy()
    ads.positions = np.asarray(ads.positions, dtype=float) @ _rotz(theta_deg).T
    ads.positions = np.asarray(ads.positions, dtype=float) + np.asarray([0.0, 0.0, 1.95], dtype=float)
    return slab + ads


def _sorted_h_surface_min_distances(frame: Atoms, *, slab_n: int) -> np.ndarray:
    dmat = np.asarray(frame.get_all_distances(mic=False), dtype=float)
    out = []
    for atom_idx in range(int(slab_n) + 1, len(frame)):
        out.append(float(np.min(dmat[int(atom_idx), : int(slab_n)])))
    return np.sort(np.asarray(out, dtype=float))


class TestBasinMappingRobustness(unittest.TestCase):
    def test_binding_surface_distance_merges_micro_perturbed_same_basin(self):
        frame_a = _co_frame(shift=[0.0, 0.0, 0.0])
        frame_b = _co_frame(shift=[0.02, -0.01, 0.01])
        slab_n = len(_anisotropic_support())
        self.assertEqual(build_binding_pairs(frame_a, slab_n=slab_n, binding_tau=1.15), [(0, 0)])
        self.assertEqual(build_binding_pairs(frame_b, slab_n=slab_n, binding_tau=1.15), [(0, 0)])

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_a, frame_b],
            energies=np.asarray([0.0, 0.01], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="binding_only",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="greedy",
        )

        self.assertEqual(len(basins), 1)
        self.assertEqual(basins[0]["member_candidate_ids"], [0, 1])
        self.assertEqual(meta["surface_atom_mode"], "binding_only")

    def test_basin_builder_binding_surface_distance_merges_micro_perturbed_same_basin(self):
        slab = _anisotropic_support()
        ads = _co_adsorbate()
        frame_a = _co_frame(shift=[0.0, 0.0, 0.0])
        frame_b = _co_frame(shift=[0.02, -0.01, 0.01])

        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                dedup_metric="binding_surface_distance",
                energy_window_ev=1.0,
                desorption_min_bonds=1,
                surface_reconstruction_enabled=False,
                work_dir=Path(td),
            )
            out = BasinBuilder(config=cfg).build(
                frames=[frame_a, frame_b],
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=2,
            )

        self.assertEqual(out.summary["n_basins"], 1)
        self.assertEqual(out.basins[0].member_candidate_ids, [0, 1])

    def test_binding_surface_distance_currently_collapses_same_anchor_rotated_nh3_on_anisotropic_support(self):
        frame_a = _nh3_frame(0.0)
        frame_b = _nh3_frame(55.0)
        slab_n = len(_anisotropic_support())

        self.assertEqual(build_binding_pairs(frame_a, slab_n=slab_n, binding_tau=1.15), [(0, 0)])
        self.assertEqual(build_binding_pairs(frame_b, slab_n=slab_n, binding_tau=1.15), [(0, 0)])

        h_env_a = _sorted_h_surface_min_distances(frame_a, slab_n=slab_n)
        h_env_b = _sorted_h_surface_min_distances(frame_b, slab_n=slab_n)
        self.assertGreater(float(np.max(np.abs(h_env_a - h_env_b))), 0.05)

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_a, frame_b],
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

        # This test documents a current robustness gap:
        # the default binding-surface descriptor sees only the bound N atom, so
        # azimuth-distinct NH3 states on an anisotropic support collapse.
        self.assertEqual(len(basins), 1)
        self.assertEqual(basins[0]["member_candidate_ids"], [0, 1])
        self.assertEqual(meta["surface_atom_mode"], "binding_only")

    def test_basin_builder_currently_collapses_same_anchor_rotated_nh3_on_anisotropic_support(self):
        slab = _anisotropic_support()
        ads = _nh3_adsorbate()
        frame_a = _nh3_frame(0.0)
        frame_b = _nh3_frame(55.0)

        with tempfile.TemporaryDirectory() as td:
            cfg = BasinConfig(
                dedup_metric="binding_surface_distance",
                energy_window_ev=1.0,
                desorption_min_bonds=1,
                surface_reconstruction_enabled=False,
                work_dir=Path(td),
            )
            out = BasinBuilder(config=cfg).build(
                frames=[frame_a, frame_b],
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=2,
            )

        self.assertEqual(out.summary["n_basins"], 1)
        self.assertEqual(out.basins[0].member_candidate_ids, [0, 1])
        self.assertEqual(out.summary["dedup_meta"]["surface_atom_mode"], "binding_only")

    def test_binding_surface_distance_can_separate_rotated_nh3_case_when_all_adsorbate_atoms_are_used(self):
        frame_a = _nh3_frame(0.0)
        frame_b = _nh3_frame(55.0)
        slab_n = len(_anisotropic_support())

        basins, meta = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_a, frame_b],
            energies=np.asarray([0.0, 0.02], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="all",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="greedy",
        )

        self.assertEqual(len(basins), 2)
        self.assertEqual(sorted(b["member_candidate_ids"] for b in basins), [[0], [1]])
        self.assertEqual(meta["surface_atom_mode"], "all")

    def test_binding_surface_distance_merges_symmetry_equivalent_rotated_nh3_on_threefold_support(self):
        frame_a = _nh3_frame_on_threefold_support(0.0)
        frame_b = _nh3_frame_on_threefold_support(120.0)
        slab_n = len(_threefold_support())

        self.assertEqual(build_binding_pairs(frame_a, slab_n=slab_n, binding_tau=1.15), [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(build_binding_pairs(frame_b, slab_n=slab_n, binding_tau=1.15), [(0, 0), (0, 1), (0, 2)])

        basins_binding_only, meta_binding_only = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_a, frame_b],
            energies=np.asarray([0.0, 0.01], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="binding_only",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="greedy",
        )
        basins_all, meta_all = cluster_by_binding_pattern_and_surface_distance(
            frames=[frame_a, frame_b],
            energies=np.asarray([0.0, 0.01], dtype=float),
            slab_n=slab_n,
            binding_tau=1.15,
            surface_distance_threshold=0.30,
            surface_nearest_k=8,
            surface_atom_mode="all",
            surface_relative=False,
            surface_rmsd_gate=0.25,
            cluster_method="greedy",
        )

        self.assertEqual(len(basins_binding_only), 1)
        self.assertEqual(basins_binding_only[0]["member_candidate_ids"], [0, 1])
        self.assertEqual(meta_binding_only["surface_atom_mode"], "binding_only")
        self.assertEqual(len(basins_all), 1)
        self.assertEqual(basins_all[0]["member_candidate_ids"], [0, 1])
        self.assertEqual(meta_all["surface_atom_mode"], "all")


if __name__ == "__main__":
    unittest.main()
