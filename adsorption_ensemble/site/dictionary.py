from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from ase import Atoms

from .primitives import SitePrimitive


def build_site_dictionary(primitives: list[SitePrimitive], slab: Atoms | None = None) -> dict[str, Any]:
    sites: dict[str, dict[str, Any]] = {}
    kinds: dict[str, list[str]] = {"1c": [], "2c": [], "3c": [], "4c": []}
    basis_groups: dict[str, list[str]] = {}
    for idx, p in enumerate(primitives):
        site_id = f"site_{idx:05d}"
        basis_id = -1 if p.basis_id is None else int(p.basis_id)
        basis_key = f"basis_{basis_id:05d}"
        topology = str(p.site_label) if p.site_label is not None and str(p.site_label).strip() else str(p.kind)
        record = {
            "site_id": site_id,
            "kind": p.kind,
            "atom_ids": [int(i) for i in p.atom_ids],
            "center": _to_list(p.center),
            "normal": _to_list(p.normal),
            "t1": _to_list(p.t1),
            "t2": _to_list(p.t2),
            "topo_hash": p.topo_hash,
            "basis_id": basis_id,
            "site_label": (None if p.site_label is None else str(p.site_label)),
            "embedding": None if p.embedding is None else _to_list(p.embedding),
            # AutoAdsorbate-style alias fields for easier cross-library review.
            "coordinates": _to_list(p.center),
            "connectivity": int(len(p.atom_ids)),
            "topology": topology,
            "n_vector": _to_list(p.normal),
            "h_vector": _to_list(p.t1),
            "site_formula": _site_formula(p.atom_ids, slab),
        }
        sites[site_id] = record
        kinds.setdefault(p.kind, []).append(site_id)
        basis_groups.setdefault(basis_key, []).append(site_id)
    meta = {
        "n_sites": len(primitives),
        "n_kinds": {k: len(v) for k, v in kinds.items()},
        "n_basis_groups": len(basis_groups),
    }
    return {"meta": meta, "sites": sites, "kinds": kinds, "basis_groups": basis_groups}


def _to_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(arr, dtype=float).reshape(-1).tolist()]


def _site_formula(atom_ids: tuple[int, ...], slab: Atoms | None) -> str:
    if slab is None:
        return f"{int(len(atom_ids))}c"
    symbols = []
    for idx in atom_ids:
        i = int(idx)
        if 0 <= i < len(slab):
            symbols.append(str(slab[i].symbol))
    if not symbols:
        return f"{int(len(atom_ids))}c"
    counts = Counter(symbols)
    parts = []
    for sym in sorted(counts):
        n = int(counts[sym])
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)
