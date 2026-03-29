import unittest

from ase.build import fcc111, molecule

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

