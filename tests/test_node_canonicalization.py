import unittest

import numpy as np
from ase.build import fcc111, molecule

from adsorption_ensemble.basin.dedup import binding_signature
from adsorption_ensemble.basin.types import Basin
from adsorption_ensemble.node import NodeConfig, basin_to_node


class TestNodeCanonicalization(unittest.TestCase):
    def test_node_id_invariant_to_adsorbate_permutation(self):
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        ads = molecule("H2O")
        frame = slab + ads
        b1 = Basin(
            basin_id=0,
            atoms=frame,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[],
            denticity=0,
            signature="none",
        )
        node1 = basin_to_node(b1, slab_n=len(slab), cfg=NodeConfig(bond_tau=1.25, node_hash_len=24))
        perm = [2, 0, 1]
        ads_p = ads[perm]
        frame_p = slab + ads_p
        b2 = Basin(
            basin_id=0,
            atoms=frame_p,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[],
            denticity=0,
            signature="none",
        )
        node2 = basin_to_node(b2, slab_n=len(slab), cfg=NodeConfig(bond_tau=1.25, node_hash_len=24))
        self.assertEqual(node1.node_id, node2.node_id)

    def test_node_id_changes_with_binding_graph(self):
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        b1 = Basin(
            basin_id=0,
            atoms=frame,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[],
            denticity=0,
            signature="none",
        )
        b2 = Basin(
            basin_id=1,
            atoms=frame,
            energy_ev=0.0,
            member_candidate_ids=[1],
            binding_pairs=[(0, 0)],
            denticity=1,
            signature="none",
        )
        cfg = NodeConfig(bond_tau=1.25, node_hash_len=24)
        n1 = basin_to_node(b1, slab_n=len(slab), cfg=cfg)
        n2 = basin_to_node(b2, slab_n=len(slab), cfg=cfg)
        self.assertNotEqual(n1.node_id, n2.node_id)

    def test_current_node_id_depends_on_absolute_surface_atom_ids_for_equivalent_top_sites(self):
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 2)

        ads = molecule("CO")
        cfg = NodeConfig(bond_tau=1.25, node_hash_len=24)

        frame_a = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] += slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.8])
        frame_b = slab.copy() + ads.copy()
        frame_b.positions[len(slab) :] += slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.8])

        b1 = Basin(
            basin_id=0,
            atoms=frame_a,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[(0, int(top_ids[0]))],
            denticity=1,
            signature="none",
        )
        b2 = Basin(
            basin_id=1,
            atoms=frame_b,
            energy_ev=0.0,
            member_candidate_ids=[1],
            binding_pairs=[(0, int(top_ids[1]))],
            denticity=1,
            signature="none",
        )

        node1 = basin_to_node(b1, slab_n=len(slab), cfg=cfg)
        node2 = basin_to_node(b2, slab_n=len(slab), cfg=cfg)
        self.assertNotEqual(node1.node_id, node2.node_id)
        self.assertEqual(
            binding_signature(
                b1.binding_pairs,
                frame=frame_a,
                slab_n=len(slab),
                surface_reference=slab,
                mode="reference_canonical",
            ),
            binding_signature(
                b2.binding_pairs,
                frame=frame_b,
                slab_n=len(slab),
                surface_reference=slab,
                mode="reference_canonical",
            ),
        )

    def test_current_node_id_splits_equivalent_bidentate_top_pairs(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 4)

        pair_buckets: dict[float, list[tuple[int, int]]] = {}
        for idx_a, a in enumerate(top_ids):
            for b in top_ids[idx_a + 1 :]:
                d = float(slab.get_distance(int(a), int(b), mic=True))
                key = round(d, 6)
                pair_buckets.setdefault(key, []).append((int(a), int(b)))

        selected_pairs = None
        for pairs in pair_buckets.values():
            if len(pairs) < 2:
                continue
            for i, p0 in enumerate(pairs):
                for p1 in pairs[i + 1 :]:
                    if set(p0).isdisjoint(set(p1)):
                        selected_pairs = (p0, p1)
                        break
                if selected_pairs is not None:
                    break
            if selected_pairs is not None:
                break
        self.assertIsNotNone(selected_pairs)
        pair_a, pair_b = selected_pairs

        ads = molecule("CO")
        cfg = NodeConfig(bond_tau=1.25, node_hash_len=24)

        frame_a = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] += np.mean(slab.positions[list(pair_a)], axis=0) + np.array([0.0, 0.0, 1.8])
        frame_b = slab.copy() + ads.copy()
        frame_b.positions[len(slab) :] += np.mean(slab.positions[list(pair_b)], axis=0) + np.array([0.0, 0.0, 1.8])

        b1 = Basin(
            basin_id=0,
            atoms=frame_a,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[(0, int(pair_a[0])), (1, int(pair_a[1]))],
            denticity=2,
            signature="none",
        )
        b2 = Basin(
            basin_id=1,
            atoms=frame_b,
            energy_ev=0.0,
            member_candidate_ids=[1],
            binding_pairs=[(0, int(pair_b[0])), (1, int(pair_b[1]))],
            denticity=2,
            signature="none",
        )

        node1 = basin_to_node(b1, slab_n=len(slab), cfg=cfg)
        node2 = basin_to_node(b2, slab_n=len(slab), cfg=cfg)
        self.assertNotEqual(node1.node_id, node2.node_id)
        self.assertEqual(
            binding_signature(
                b1.binding_pairs,
                frame=frame_a,
                slab_n=len(slab),
                surface_reference=slab,
                mode="reference_canonical",
            ),
            binding_signature(
                b2.binding_pairs,
                frame=frame_b,
                slab_n=len(slab),
                surface_reference=slab,
                mode="reference_canonical",
            ),
        )

    def test_surface_geometry_identity_mode_merges_equivalent_top_sites(self):
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 2)

        ads = molecule("CO")
        cfg = NodeConfig(bond_tau=1.25, node_hash_len=24, node_identity_mode="surface_geometry")

        frame_a = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] += slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.8])
        frame_b = slab.copy() + ads.copy()
        frame_b.positions[len(slab) :] += slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.8])

        b1 = Basin(
            basin_id=0,
            atoms=frame_a,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[(0, int(top_ids[0]))],
            denticity=1,
            signature="none",
        )
        b2 = Basin(
            basin_id=1,
            atoms=frame_b,
            energy_ev=0.0,
            member_candidate_ids=[1],
            binding_pairs=[(0, int(top_ids[1]))],
            denticity=1,
            signature="none",
        )

        node1 = basin_to_node(b1, slab_n=len(slab), cfg=cfg, surface_reference=slab)
        node2 = basin_to_node(b2, slab_n=len(slab), cfg=cfg, surface_reference=slab)
        self.assertEqual(node1.node_id, node2.node_id)
        self.assertEqual(node1.surface_env_key, node2.surface_env_key)
        self.assertEqual(node1.surface_geometry_key, node2.surface_geometry_key)
        self.assertNotEqual(node1.node_id_legacy, node2.node_id_legacy)

    def test_surface_geometry_identity_mode_merges_equivalent_bidentate_top_pairs(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - np.max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 4)

        pair_buckets: dict[float, list[tuple[int, int]]] = {}
        for idx_a, a in enumerate(top_ids):
            for b in top_ids[idx_a + 1 :]:
                d = float(slab.get_distance(int(a), int(b), mic=True))
                key = round(d, 6)
                pair_buckets.setdefault(key, []).append((int(a), int(b)))

        selected_pairs = None
        for pairs in pair_buckets.values():
            if len(pairs) < 2:
                continue
            for i, p0 in enumerate(pairs):
                for p1 in pairs[i + 1 :]:
                    if set(p0).isdisjoint(set(p1)):
                        selected_pairs = (p0, p1)
                        break
                if selected_pairs is not None:
                    break
            if selected_pairs is not None:
                break
        self.assertIsNotNone(selected_pairs)
        pair_a, pair_b = selected_pairs

        ads = molecule("CO")
        cfg = NodeConfig(bond_tau=1.25, node_hash_len=24, node_identity_mode="surface_geometry")

        frame_a = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] += np.mean(slab.positions[list(pair_a)], axis=0) + np.array([0.0, 0.0, 1.8])
        frame_b = slab.copy() + ads.copy()
        frame_b.positions[len(slab) :] += np.mean(slab.positions[list(pair_b)], axis=0) + np.array([0.0, 0.0, 1.8])

        b1 = Basin(
            basin_id=0,
            atoms=frame_a,
            energy_ev=0.0,
            member_candidate_ids=[0],
            binding_pairs=[(0, int(pair_a[0])), (1, int(pair_a[1]))],
            denticity=2,
            signature="none",
        )
        b2 = Basin(
            basin_id=1,
            atoms=frame_b,
            energy_ev=0.0,
            member_candidate_ids=[1],
            binding_pairs=[(0, int(pair_b[0])), (1, int(pair_b[1]))],
            denticity=2,
            signature="none",
        )

        node1 = basin_to_node(b1, slab_n=len(slab), cfg=cfg, surface_reference=slab)
        node2 = basin_to_node(b2, slab_n=len(slab), cfg=cfg, surface_reference=slab)
        self.assertEqual(node1.node_id, node2.node_id)
        self.assertEqual(node1.surface_geometry_key, node2.surface_geometry_key)
        self.assertNotEqual(node1.node_id_legacy, node2.node_id_legacy)
