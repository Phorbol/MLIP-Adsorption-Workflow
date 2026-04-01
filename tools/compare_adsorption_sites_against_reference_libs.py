from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bcc100, bcc110, bcc111, bulk, fcc100, fcc110, fcc111, fcc211, hcp0001, hcp10m10, surface

from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import SurfacePreprocessor


def _build_slabs() -> dict[str, Atoms]:
    return {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc110": fcc110("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=10.0),
        "bcc100": bcc100("Fe", size=(4, 4, 4), vacuum=10.0),
        "bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=10.0),
        "bcc111": bcc111("Fe", size=(4, 4, 4), vacuum=10.0),
        "hcp0001": hcp0001("Ru", size=(4, 4, 4), vacuum=10.0),
        "hcp10m10": hcp10m10("Ru", size=(4, 4, 4), vacuum=10.0),
        "cu321": surface(bulk("Cu", "fcc", a=3.6, cubic=True), (3, 2, 1), layers=4, vacuum=10.0).repeat((2, 2, 1)),
    }


def _mic_2d_distance(frac_a: np.ndarray, frac_b: np.ndarray) -> float:
    d = np.asarray(frac_a, dtype=float) - np.asarray(frac_b, dtype=float)
    d -= np.round(d)
    return float(np.linalg.norm(d))


def _fractional_2d_positions(slab: Atoms, primitives: list) -> dict[str, list[np.ndarray]]:
    cell = np.asarray(slab.cell.array, dtype=float)
    out: dict[str, list[np.ndarray]] = {}
    for p in primitives:
        frac = np.linalg.solve(cell.T, np.asarray(p.center, dtype=float))
        uv = np.asarray([frac[0] % 1.0, frac[1] % 1.0], dtype=float)
        out.setdefault(str(p.kind), []).append(uv)
    return out


def _run_our_site_extraction(slab: Atoms) -> dict:
    ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
    raw = PrimitiveBuilder().build(slab, ctx)
    z = slab.get_atomic_numbers().astype(float)
    atom_features = (z / (np.max(z) + 1e-12)).reshape(-1, 1)
    emb = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.22)).fit_transform(
        slab=slab,
        primitives=list(raw),
        atom_features=atom_features,
    )
    raw_counts = Counter([p.kind for p in emb.primitives])
    basis_counts = Counter([p.kind for p in emb.basis_primitives])
    basis_uv = _fractional_2d_positions(slab, emb.basis_primitives)
    return {
        "n_surface_atoms": int(len(ctx.detection.surface_atom_ids)),
        "raw_counts": dict(raw_counts),
        "basis_counts": dict(basis_counts),
        "basis_uv": {k: [v.tolist() for v in vals] for k, vals in basis_uv.items()},
    }


def _ase_reference_check(slab: Atoms, ours: dict) -> dict:
    info = slab.info.get("adsorbate_info", {})
    sites = info.get("sites", {}) if isinstance(info, dict) else {}
    if not isinstance(sites, dict) or not sites:
        return {"status": "no_reference_sites"}
    basis_uv = {k: [np.asarray(x, dtype=float) for x in vals] for k, vals in ours["basis_uv"].items()}
    name_to_kind = {
        "ontop": "1c",
        "bridge": "2c",
        "shortbridge": "2c",
        "longbridge": "2c",
        "fcc": "3c",
        "hcp": "3c",
        "hollow": None,
    }
    basis_counts = Counter({str(k): int(v) for k, v in ours["basis_counts"].items()})
    expected = {"ontop": 0, "bridge": 0, "hollow": 0}
    for name in sites.keys():
        lname = str(name).lower()
        if lname == "ontop":
            expected["ontop"] += 1
        elif "bridge" in lname:
            expected["bridge"] += 1
        elif lname in {"fcc", "hcp", "hollow"}:
            expected["hollow"] += 1
    observed = {
        "ontop": int(basis_counts.get("1c", 0)),
        "bridge": int(basis_counts.get("2c", 0)),
        "hollow": int(basis_counts.get("3c", 0) + basis_counts.get("4c", 0)),
    }
    strict_match = bool(observed == expected)
    site_dist = {}
    for name, uv in sites.items():
        kind = name_to_kind.get(str(name).lower())
        if str(name).lower() == "hollow":
            # Hollow can correspond to 3c or 4c depending on lattice family.
            cand = basis_uv.get("4c", []) + basis_uv.get("3c", [])
            target = np.asarray(uv[:2], dtype=float)
            site_dist[str(name)] = (None if not cand else float(min(_mic_2d_distance(target, c) for c in cand)))
            continue
        if kind is None:
            continue
        cand = basis_uv.get(kind, [])
        if not cand:
            site_dist[str(name)] = None
            continue
        target = np.asarray(uv[:2], dtype=float)
        dmin = min(_mic_2d_distance(target, c) for c in cand)
        site_dist[str(name)] = float(dmin)
    return {
        "status": "ok",
        "site_min_mic_uv_distance": site_dist,
        "strict_mapped_counts_expected": expected,
        "strict_mapped_counts_observed": observed,
        "strict_mapped_counts_match": strict_match,
    }


def _pymatgen_reference_check(slab: Atoms, ours: dict) -> dict:
    try:
        from pymatgen.analysis.adsorption import AdsorbateSiteFinder
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception as exc:
        return {"status": "import_error", "error": f"{type(exc).__name__}: {exc}"}
    structure = AseAtomsAdaptor.get_structure(slab)
    asf = AdsorbateSiteFinder(structure)
    out = {}
    for name in ("ontop", "bridge", "hollow"):
        try:
            ref = asf.find_adsorption_sites(positions=[name], symm_reduce=0.01)
            out[name] = int(len(ref.get("all", [])))
        except Exception as exc:
            out[name] = f"error:{type(exc).__name__}"
    basis = ours["basis_counts"]
    our_hollow = int(basis.get("3c", 0)) + int(basis.get("4c", 0))
    return {
        "status": "ok",
        "pymatgen_counts": out,
        "our_counts_mapped": {
            "ontop": int(basis.get("1c", 0)),
            "bridge": int(basis.get("2c", 0)),
            "hollow": int(our_hollow),
        },
    }


def _dockonsurf_check() -> dict:
    try:
        import dockonsurf  # type: ignore

        version = getattr(dockonsurf, "__version__", "unknown")
        return {"status": "ok", "version": str(version)}
    except Exception as exc:
        return {"status": "import_error", "error": f"{type(exc).__name__}: {exc}"}


def _supercell_equivalence_check() -> dict:
    out = []
    for n in (3, 4, 5):
        slab = fcc111("Pt", size=(n, n, 4), vacuum=10.0)
        ours = _run_our_site_extraction(slab)
        out.append({"size": [n, n, 4], "basis_counts": ours["basis_counts"]})
    return {"series": out}


def main() -> int:
    out_dir = Path("artifacts/autoresearch/physics_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    slabs = _build_slabs()
    rows = []
    for name, slab in slabs.items():
        ours = _run_our_site_extraction(slab)
        rows.append(
            {
                "slab": name,
                "ours": ours,
                "ase_reference": _ase_reference_check(slab, ours),
                "pymatgen_reference": _pymatgen_reference_check(slab, ours),
            }
        )
    payload = {
        "rows": rows,
        "dockonsurf": _dockonsurf_check(),
        "fcc111_supercell_equivalence": _supercell_equivalence_check(),
    }
    ase_total = int(sum(1 for r in rows if r["ase_reference"].get("status") == "ok"))
    ase_matched = int(
        sum(
            1
            for r in rows
            if r["ase_reference"].get("status") == "ok" and bool(r["ase_reference"].get("strict_mapped_counts_match"))
        )
    )
    payload["ase_strict_match"] = {"matched": ase_matched, "total": ase_total, "ratio": float(ase_matched / max(1, ase_total))}
    out_path = out_dir / "site_reference_comparison.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
