from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from .pipeline import SurfaceContext


def export_surface_detection_report(slab: Atoms, context: SurfaceContext, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = context.detection.surface_atom_ids
    scores = context.detection.exposure_scores
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()

    csv_path = out / "surface_atoms.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("index,symbol,x,y,z,exposure_score\n")
        for i in ids:
            x, y, z = positions[i]
            f.write(f"{i},{symbols[i]},{x:.8f},{y:.8f},{z:.8f},{float(scores[i]):.8f}\n")

    xyz_path = out / "surface_atoms_only.xyz"
    surface_atoms = slab[ids]
    write(xyz_path, surface_atoms)

    tagged_path = out / "slab_with_surface_tags.extxyz"
    tagged = slab.copy()
    tags = np.zeros(len(tagged), dtype=int)
    tags[ids] = 1
    tagged.set_tags(tags.tolist())
    write(tagged_path, tagged)

    return {
        "surface_csv": csv_path,
        "surface_xyz": xyz_path,
        "tagged_slab": tagged_path,
    }
