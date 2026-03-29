from __future__ import annotations

from typing import Any

import numpy as np

from .primitives import SitePrimitive


def build_site_dictionary(primitives: list[SitePrimitive]) -> dict[str, Any]:
    sites: dict[str, dict[str, Any]] = {}
    kinds: dict[str, list[str]] = {"1c": [], "2c": [], "3c": [], "4c": []}
    basis_groups: dict[str, list[str]] = {}
    for idx, p in enumerate(primitives):
        site_id = f"site_{idx:05d}"
        basis_id = -1 if p.basis_id is None else int(p.basis_id)
        basis_key = f"basis_{basis_id:05d}"
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
            "embedding": None if p.embedding is None else _to_list(p.embedding),
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
