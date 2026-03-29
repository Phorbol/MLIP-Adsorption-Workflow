from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read


def read_molecule_any(input_path: str | Path) -> Atoms:
    p = Path(input_path)
    try:
        atoms = read(p.as_posix())
        if len(atoms) > 0:
            return atoms
    except Exception:
        pass
    ext = p.suffix.lower()
    if ext in {".gjf", ".com"}:
        return _read_gaussian_input(p)
    raise ValueError(f"Unsupported molecular input file: {p}")


def _read_gaussian_input(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    charge_idx = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                int(parts[0])
                int(parts[1])
                charge_idx = i
                break
            except ValueError:
                continue
    if charge_idx is None:
        raise ValueError(f"Cannot locate charge/multiplicity line in Gaussian input: {path}")
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[charge_idx + 1 :]:
        s = line.strip()
        if not s:
            break
        cols = s.split()
        if len(cols) < 4:
            break
        sym = cols[0]
        try:
            x = float(cols[1])
            y = float(cols[2])
            z = float(cols[3])
        except ValueError:
            break
        symbols.append(sym)
        positions.append([x, y, z])
    if not symbols:
        raise ValueError(f"No atomic coordinates parsed from Gaussian input: {path}")
    return Atoms(symbols=symbols, positions=positions)
