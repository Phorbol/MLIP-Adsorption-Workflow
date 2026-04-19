import unittest
from unittest import mock

import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption_ensemble.basin.dedup import (
    cluster_by_signature_and_mace_node_l2,
    merge_basin_representatives_by_mace_node_l2,
)


class TestBasinDedupMaceNodeL2(unittest.TestCase):
    def test_cluster_mace_node_l2_is_invariant_to_equivalent_adsorbate_atom_permutation(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads_a = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 1.8],
                [0.94, 0.0, 2.4],
                [-0.47, 0.81, 2.4],
                [-0.47, -0.81, 2.4],
            ],
        )
        ads_b = ads_a[[0, 2, 3, 1]]
        frame_a = slab + ads_a
        frame_b = slab + ads_b
        frames = [frame_a, frame_b]
        energies = np.asarray([0.0, 0.01], dtype=float)

        n = len(frame_a)
        d = 3
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        x0[len(slab) :] = np.asarray(
            [
                [10.0, 0.0, 0.0],  # N
                [1.0, 0.0, 0.0],   # H1
                [2.0, 0.0, 0.0],   # H2
                [3.0, 0.0, 0.0],   # H3
            ],
            dtype=float,
        )
        x1[len(slab) :] = np.asarray(
            [
                [10.0, 0.0, 0.0],  # N
                [2.0, 0.0, 0.0],   # permuted H2
                [3.0, 0.0, 0.0],   # permuted H3
                [1.0, 0.0, 0.0],   # permuted H1
            ],
            dtype=float,
        )

        basins, meta = cluster_by_signature_and_mace_node_l2(
            frames=frames,
            energies=energies,
            slab_n=len(slab),
            binding_tau=0.0,
            node_l2_threshold=1.0e-8,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float64",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
            signature_mode="none",
            use_signature_grouping=False,
            l2_mode="sum",
        )
        self.assertEqual(len(basins), 1)
        self.assertEqual(meta["l2_mode"], "sum")

    def test_cluster_merges_by_node_l2(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        frames = [frame.copy(), frame.copy(), frame.copy()]
        energies = np.asarray([0.0, 0.1, 0.2], dtype=float)
        n = len(frame)
        d = 8
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        x2 = np.zeros((n, d), dtype=float)
        x2[:, 0] = 10.0
        basins, meta = cluster_by_signature_and_mace_node_l2(
            frames=frames,
            energies=energies,
            slab_n=len(slab),
            binding_tau=0.0,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1, x2],
        )
        self.assertTrue(bool(meta))
        self.assertEqual(len(basins), 2)
        member_sizes = sorted(len(b["member_candidate_ids"]) for b in basins)
        self.assertEqual(member_sizes, [1, 2])

    def test_hierarchical_and_fuzzy_can_merge_chain_without_k(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        frames = [frame.copy(), frame.copy(), frame.copy()]
        energies = np.asarray([0.0, 0.1, 0.2], dtype=float)
        n = len(frame)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        x2 = np.zeros((n, d), dtype=float)
        x1[:, 0] = 0.19
        x2[:, 0] = 0.38
        for method in ("hierarchical", "fuzzy"):
            basins, _ = cluster_by_signature_and_mace_node_l2(
                frames=frames,
                energies=energies,
                slab_n=len(slab),
                binding_tau=0.0,
                node_l2_threshold=0.2,
                mace_model_path=None,
                mace_device="cpu",
                mace_dtype="float32",
                mace_max_edges_per_batch=1000,
                mace_layers_to_keep=-1,
                mace_head_name=None,
                mace_mlp_energy_key=None,
                cluster_method=method,
                l2_mode="mean_atom",
                node_descriptors=[x0, x1, x2],
            )
            self.assertEqual(len(basins), 1, msg=f"method={method}")

    def test_pure_mace_mode_can_merge_across_signatures(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        frames = [frame.copy(), frame.copy()]
        energies = np.asarray([0.0, 0.1], dtype=float)
        n = len(frame)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        basins, _ = cluster_by_signature_and_mace_node_l2(
            frames=frames,
            energies=energies,
            slab_n=len(slab),
            binding_tau=0.0,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
            signature_mode="absolute",
            use_signature_grouping=False,
        )
        self.assertEqual(len(basins), 1)
        self.assertNotEqual(str(basins[0]["signature"]), "all")

    def test_cluster_forwards_mace_enable_cueq_to_inferencer(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        frames = [frame.copy(), frame.copy()]
        energies = np.asarray([0.0, 0.1], dtype=float)
        captured = {}

        class _DummyInferencer:
            def __init__(self, cfg):
                captured["cfg"] = cfg

            def infer_node_descriptors(self, frames_in):
                n = len(frames_in[0])
                d = 4
                desc = [np.zeros((n, d), dtype=float) for _ in frames_in]
                return desc, np.zeros((len(frames_in),), dtype=float), {"provided_by_dummy": True}

        with (
            mock.patch(
                "adsorption_ensemble.relax.backends.normalize_mace_descriptor_config",
                return_value=("/root/.cache/mace/mace-omat-0-small.model", "cuda", "float32"),
            ),
            mock.patch(
                "adsorption_ensemble.conformer_md.mace_inference.MACEBatchInferencer",
                _DummyInferencer,
            ),
        ):
            basins, meta = cluster_by_signature_and_mace_node_l2(
                frames=frames,
                energies=energies,
                slab_n=len(slab),
                binding_tau=0.0,
                node_l2_threshold=1e-6,
                mace_model_path="/root/.cache/mace/mace-omat-0-small.model",
                mace_device="cuda",
                mace_dtype="float32",
                mace_max_edges_per_batch=1000,
                mace_layers_to_keep=-1,
                mace_head_name=None,
                mace_mlp_energy_key=None,
                mace_enable_cueq=True,
            )
        self.assertEqual(len(basins), 1)
        self.assertTrue(bool(meta["enable_cueq"]))
        self.assertTrue(bool(captured["cfg"].enable_cueq))

    def test_final_basin_merge_can_collapse_translated_duplicate_representatives(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        basins = [
            {
                "basin_id": 0,
                "atoms": frame.copy(),
                "energy": 0.0,
                "member_candidate_ids": [0, 2],
                "binding_pairs": [(0, 0)],
                "signature": "sig_top_a",
            },
            {
                "basin_id": 1,
                "atoms": frame.copy(),
                "energy": 0.05,
                "member_candidate_ids": [1, 3],
                "binding_pairs": [(0, 1)],
                "signature": "sig_top_b",
            },
        ]
        n = len(frame)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        merged, meta = merge_basin_representatives_by_mace_node_l2(
            basins=basins,
            slab_n=len(slab),
            binding_tau=1.15,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
        )
        self.assertEqual(meta["n_input_basins"], 2)
        self.assertEqual(meta["n_output_basins"], 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(sorted(merged[0]["member_candidate_ids"]), [0, 1, 2, 3])
        self.assertEqual(sorted(merged[0]["source_basin_ids"]), [0, 1])

    def test_final_basin_merge_respects_energy_gate(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        basins = [
            {
                "basin_id": 0,
                "atoms": frame.copy(),
                "energy": 0.0,
                "member_candidate_ids": [0],
                "binding_pairs": [(0, 0)],
                "signature": "sig_top_a",
            },
            {
                "basin_id": 1,
                "atoms": frame.copy(),
                "energy": 0.30,
                "member_candidate_ids": [1],
                "binding_pairs": [(0, 1)],
                "signature": "sig_top_b",
            },
        ]
        n = len(frame)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        merged, meta = merge_basin_representatives_by_mace_node_l2(
            basins=basins,
            slab_n=len(slab),
            binding_tau=1.15,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
            energy_gate_ev=0.05,
        )
        self.assertEqual(meta["n_input_basins"], 2)
        self.assertEqual(meta["n_output_basins"], 2)
        self.assertAlmostEqual(meta["energy_gate_ev"], 0.05)
        self.assertEqual(len(merged), 2)

    def test_final_basin_merge_can_respect_canonical_signature_grouping(self):
        frame_pt = Atoms(
            symbols=["Pt", "Cu", "C", "O"],
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.8],
                [0.0, 0.0, 2.9],
            ],
        )
        frame_cu = frame_pt.copy()
        frame_cu.positions[2:] += np.asarray([2.0, 0.0, 0.0], dtype=float)
        basins = [
            {
                "basin_id": 0,
                "atoms": frame_pt,
                "energy": 0.0,
                "member_candidate_ids": [0],
                "binding_pairs": [(0, 0)],
                "signature": "sig_pt",
            },
            {
                "basin_id": 1,
                "atoms": frame_cu,
                "energy": 0.05,
                "member_candidate_ids": [1],
                "binding_pairs": [(0, 1)],
                "signature": "sig_cu",
            },
        ]
        n = len(frame_pt)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        merged, meta = merge_basin_representatives_by_mace_node_l2(
            basins=basins,
            slab_n=2,
            binding_tau=1.15,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
            signature_mode="canonical",
            use_signature_grouping=True,
        )
        self.assertEqual(meta["n_input_basins"], 2)
        self.assertEqual(meta["n_output_basins"], 2)
        self.assertEqual(meta["signature_mode"], "canonical")
        self.assertTrue(bool(meta["use_signature_grouping"]))
        self.assertEqual(len(merged), 2)

    def test_final_basin_merge_can_use_reference_canonical_grouping_for_equivalent_sites(self):
        surface_ref = Atoms(
            symbols=["Pt", "Pt", "Pt"],
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.732, 0.0],
            ],
        )
        frame_a = Atoms(
            symbols=["Pt", "Pt", "Pt", "N", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.732, 0.0],
                [0.0, 0.0, 1.8],
                [0.9, 0.0, 2.4],
                [-0.45, 0.78, 2.4],
                [-0.45, -0.78, 2.4],
            ],
        )
        frame_b = frame_a.copy()
        frame_b.positions[:3] += np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.00, 0.00],
                [0.00, -0.02, 0.00],
            ],
            dtype=float,
        )
        frame_b.positions[3:] += np.asarray([2.0, 0.0, 0.0], dtype=float)
        basins = [
            {
                "basin_id": 0,
                "atoms": frame_a,
                "energy": 0.0,
                "member_candidate_ids": [0],
                "binding_pairs": [(0, 0)],
                "signature": "sig_a",
            },
            {
                "basin_id": 1,
                "atoms": frame_b,
                "energy": 0.05,
                "member_candidate_ids": [1],
                "binding_pairs": [(0, 1)],
                "signature": "sig_b",
            },
        ]
        n = len(frame_a)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        merged, meta = merge_basin_representatives_by_mace_node_l2(
            basins=basins,
            slab_n=3,
            binding_tau=1.15,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
            signature_mode="reference_canonical",
            use_signature_grouping=True,
            surface_reference=surface_ref,
        )
        self.assertEqual(meta["n_input_basins"], 2)
        self.assertEqual(meta["n_output_basins"], 1)
        self.assertEqual(meta["signature_mode"], "reference_canonical")
        self.assertEqual(len(merged), 1)

    def test_final_basin_merge_reference_canonical_respects_inequivalent_reference_sites(self):
        surface_ref = Atoms(
            symbols=["Pt", "Cu"],
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        )
        frame_pt = Atoms(
            symbols=["Pt", "Cu", "C", "O"],
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.8],
                [0.0, 0.0, 2.9],
            ],
        )
        frame_cu = frame_pt.copy()
        frame_cu.positions[2:] += np.asarray([2.0, 0.0, 0.0], dtype=float)
        basins = [
            {
                "basin_id": 0,
                "atoms": frame_pt,
                "energy": 0.0,
                "member_candidate_ids": [0],
                "binding_pairs": [(0, 0)],
                "signature": "sig_pt",
            },
            {
                "basin_id": 1,
                "atoms": frame_cu,
                "energy": 0.05,
                "member_candidate_ids": [1],
                "binding_pairs": [(0, 1)],
                "signature": "sig_cu",
            },
        ]
        n = len(frame_pt)
        d = 4
        x0 = np.zeros((n, d), dtype=float)
        x1 = np.zeros((n, d), dtype=float)
        merged, meta = merge_basin_representatives_by_mace_node_l2(
            basins=basins,
            slab_n=2,
            binding_tau=1.15,
            node_l2_threshold=1e-6,
            mace_model_path=None,
            mace_device="cpu",
            mace_dtype="float32",
            mace_max_edges_per_batch=1000,
            mace_layers_to_keep=-1,
            mace_head_name=None,
            mace_mlp_energy_key=None,
            node_descriptors=[x0, x1],
            signature_mode="reference_canonical",
            use_signature_grouping=True,
            surface_reference=surface_ref,
        )
        self.assertEqual(meta["n_input_basins"], 2)
        self.assertEqual(meta["n_output_basins"], 2)
        self.assertEqual(meta["signature_mode"], "reference_canonical")
        self.assertEqual(len(merged), 2)
