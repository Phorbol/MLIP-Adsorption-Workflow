from __future__ import annotations

from itertools import combinations

import numpy as np
from ase import Atoms


def enumerate_primitives_delaunay(slab: Atoms, surface_atom_ids: list[int], normal_axis: int) -> dict[str, list[tuple[int, ...]]]:
    if len(surface_atom_ids) == 0:
        return {"1c": [], "2c": [], "3c": [], "4c": []}
    try:
        from scipy.spatial import Delaunay
    except Exception as exc:
        raise ImportError("scipy is required for Delaunay primitive enumeration") from exc
    tangential_axes = [ax for ax in range(3) if ax != normal_axis]
    frac = slab.get_scaled_positions(wrap=True)[surface_atom_ids]
    uv = frac[:, tangential_axes]
    if uv.shape[0] < 3:
        return {"1c": [(i,) for i in sorted(surface_atom_ids)], "2c": [], "3c": [], "4c": []}
    tri = Delaunay(uv, qhull_options="QJ")
    local_to_global = {i: gid for i, gid in enumerate(surface_atom_ids)}
    sites_1 = [(i,) for i in sorted(surface_atom_ids)]
    edges: set[tuple[int, int]] = set()
    tris: set[tuple[int, int, int]] = set()
    for simplex in tri.simplices:
        g = tuple(sorted(local_to_global[int(k)] for k in simplex))
        if len(set(g)) == 3:
            tris.add(g)
            i, j, k = g
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((i, k))))
            edges.add(tuple(sorted((j, k))))
    quads: set[tuple[int, int, int, int]] = set()
    for ti in range(len(tri.simplices)):
        for k in range(3):
            nj = int(tri.neighbors[ti, k])
            if nj <= ti:
                continue
            local_a = set(int(x) for x in tri.simplices[ti])
            local_b = set(int(x) for x in tri.simplices[nj])
            local4 = local_a.union(local_b)
            if len(local4) != 4:
                continue
            g4 = tuple(sorted(local_to_global[l] for l in local4))
            if len(set(g4)) != 4:
                continue
            quads.add(g4)
    return {"1c": sorted(sites_1), "2c": sorted(edges), "3c": sorted(tris), "4c": sorted(quads)}


def compare_graph_vs_delaunay(
    graph_sites: dict[str, list[tuple[int, ...]]],
    delaunay_sites: dict[str, list[tuple[int, ...]]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for k in ("1c", "2c", "3c", "4c"):
        g = set(tuple(x) for x in graph_sites.get(k, []))
        d = set(tuple(x) for x in delaunay_sites.get(k, []))
        out[k] = {
            "graph": len(g),
            "delaunay": len(d),
            "overlap": len(g.intersection(d)),
            "graph_only": len(g - d),
            "delaunay_only": len(d - g),
        }
    return out
