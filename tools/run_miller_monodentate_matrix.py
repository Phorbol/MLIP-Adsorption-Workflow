from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import bcc100, bcc110, bcc111, fcc100, fcc110, fcc111, fcc211, hcp0001, hcp10m10, molecule

from tools.run_ase_autoadsorbate_crosscheck import _placement_check_for_pair, _run_our_site_pipeline


def build_miller_metal_slab_suite() -> dict[str, Atoms]:
    return {
        "Pt_fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
        "Pt_fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
        "Pt_fcc110": fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
        "Pt_fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
        "Cu_fcc111": fcc111("Cu", size=(4, 4, 4), vacuum=12.0),
        "Cu_fcc100": fcc100("Cu", size=(4, 4, 4), vacuum=12.0),
        "Cu_fcc110": fcc110("Cu", size=(4, 4, 4), vacuum=12.0),
        "Fe_bcc100": bcc100("Fe", size=(4, 4, 4), vacuum=12.0),
        "Fe_bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=12.0),
        "Fe_bcc111": bcc111("Fe", size=(4, 4, 4), vacuum=12.0),
        "Ru_hcp0001": hcp0001("Ru", size=(4, 4, 4), vacuum=12.0),
        "Ru_hcp10m10": hcp10m10("Ru", size=(4, 4, 4), vacuum=12.0),
    }


def build_monodentate_suite() -> dict[str, Atoms]:
    out: dict[str, Atoms] = {"H": Atoms("H", positions=[[0.0, 0.0, 0.0]])}
    for name in ("CO", "NO", "NH3", "H2O", "CH3OH", "HCN"):
        ads = molecule(name)
        if name == "CO" and ads[0].symbol != "C":
            ads = ads[[1, 0]]
        out[name] = ads
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_miller_metal_slab_suite()
    molecules = build_monodentate_suite()
    slab_names = sorted(slabs.keys())
    mol_names = sorted(molecules.keys())
    if int(args.max_slabs) > 0:
        slab_names = slab_names[: int(args.max_slabs)]
    if int(args.max_molecules) > 0:
        mol_names = mol_names[: int(args.max_molecules)]

    slab_cache: dict[str, dict[str, Any]] = {}
    slab_rows: list[dict[str, Any]] = []
    for slab_name in slab_names:
        slab = slabs[slab_name]
        slab_dir = out_root / "slabs" / slab_name
        ours = _run_our_site_pipeline(slab, slab_dir)
        slab_cache[slab_name] = ours
        slab_rows.append(
            {
                "slab": slab_name,
                "n_atoms": int(ours["n_atoms"]),
                "n_surface_atoms": int(ours["n_surface_atoms"]),
                "n_basis_primitives": int(ours["n_basis_primitives"]),
                "basis_counts": dict(ours["basis_counts"]),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    for slab_name in slab_names:
        slab = slabs[slab_name]
        for mol_name in mol_names:
            result = _placement_check_for_pair(
                slab=slab,
                molecule=molecules[mol_name],
                site_cache=slab_cache[slab_name],
                min_accepted_distance=float(args.min_accepted_distance),
            )
            status = str(result.get("status", "exception"))
            status_counter[status] += 1
            pair_rows.append(
                {
                    "slab": slab_name,
                    "molecule": mol_name,
                    "n_ads_atoms": int(len(molecules[mol_name])),
                    **result,
                }
            )

    payload = {
        "out_root": out_root.as_posix(),
        "matrix_kind": "miller_metal_x_monodentate",
        "slabs": slab_names,
        "molecules": mol_names,
        "n_slabs": int(len(slab_names)),
        "n_molecules": int(len(mol_names)),
        "n_pairs": int(len(pair_rows)),
        "status_counts": {str(k): int(v) for k, v in sorted(status_counter.items())},
        "ok_ratio": float(status_counter.get("ok", 0) / max(1, len(pair_rows))),
        "slab_rows": slab_rows,
        "failed_pairs": [row for row in pair_rows if str(row.get("status")) != "ok"],
    }
    out_path = out_root / "miller_monodentate_matrix_summary.json"
    _write_json(out_path, payload)
    _write_json(out_root / "miller_monodentate_matrix_pair_rows.json", pair_rows)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/physics_audit/miller_monodentate_matrix")
    parser.add_argument("--max-slabs", type=int, default=0)
    parser.add_argument("--max-molecules", type=int, default=0)
    parser.add_argument("--min-accepted-distance", type=float, default=0.70)
    args = parser.parse_args()
    out_path = run(args)
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
