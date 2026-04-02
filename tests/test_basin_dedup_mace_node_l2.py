import unittest

import numpy as np
from ase.build import fcc111, molecule

from adsorption_ensemble.basin.dedup import cluster_by_signature_and_mace_node_l2


class TestBasinDedupMaceNodeL2(unittest.TestCase):
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
