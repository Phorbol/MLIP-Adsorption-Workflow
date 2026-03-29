from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms
from ase.neighborlist import build_neighbor_list, natural_cutoffs


@dataclass
class ExposedSurfaceGraph:
    surface_atom_ids: list[int]
    edges: list[tuple[int, int]]
    neighbors: dict[int, set[int]]


class ExposedSurfaceGraphBuilder:
    def __init__(self, neighbor_scale: float = 1.2):
        self.neighbor_scale = neighbor_scale

    def build(self, atoms: Atoms, surface_atom_ids: list[int]) -> ExposedSurfaceGraph:
        surface_set = set(surface_atom_ids)
        cutoffs = natural_cutoffs(atoms, mult=self.neighbor_scale)
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, bothways=True, self_interaction=False, skin=0.0)
        edges: set[tuple[int, int]] = set()
        neighbors: dict[int, set[int]] = {i: set() for i in surface_atom_ids}
        for i in surface_atom_ids:
            js, _ = nl.get_neighbors(i)
            for j in js:
                if j not in surface_set:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
                neighbors[i].add(j)
        return ExposedSurfaceGraph(surface_atom_ids=sorted(surface_atom_ids), edges=sorted(edges), neighbors=neighbors)
