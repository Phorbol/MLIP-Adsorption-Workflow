from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read


def read_molecule_any(input_path: str | Path) -> Atoms:
    p = Path(input_path)
    ext = p.suffix.lower()
    if ext in {".gjf", ".com"}:
        return _read_gaussian_input(p)
    try:
        atoms = read(p.as_posix())
        if len(atoms) > 0:
            return atoms
    except Exception:
        pass
    raise ValueError(f"Unsupported molecular input file: {p}")


def _read_gaussian_input(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    charge_idx = None
    charge = None
    multiplicity = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                charge = int(parts[0])
                multiplicity = int(parts[1])
                charge_idx = i
                break
            except ValueError:
                continue
    if charge_idx is None:
        raise ValueError(f"Cannot locate charge/multiplicity line in Gaussian input: {path}")
    symbols: list[str] = []
    positions: list[list[float]] = []
    coord_end_idx = None
    for idx, line in enumerate(lines[charge_idx + 1 :], start=charge_idx + 1):
        s = line.strip()
        if not s:
            coord_end_idx = idx
            break
        cols = s.split()
        if len(cols) < 4:
            coord_end_idx = idx
            break
        sym = cols[0]
        try:
            x = float(cols[1])
            y = float(cols[2])
            z = float(cols[3])
        except ValueError:
            coord_end_idx = idx
            break
        symbols.append(sym)
        positions.append([x, y, z])
    if not symbols:
        raise ValueError(f"No atomic coordinates parsed from Gaussian input: {path}")
    atoms = Atoms(symbols=symbols, positions=positions)
    if charge is not None:
        atoms.info["gaussian_charge"] = int(charge)
    if multiplicity is not None:
        atoms.info["gaussian_multiplicity"] = int(multiplicity)
    if coord_end_idx is None:
        coord_end_idx = charge_idx + 1 + len(symbols)
    bonds = _parse_gaussian_connectivity(lines[coord_end_idx + 1 :], n_atoms=len(symbols))
    if bonds:
        atoms.info["connectivity_bonds"] = tuple((int(i), int(j), float(order)) for i, j, order in bonds)
    return atoms


def _parse_gaussian_connectivity(lines: list[str], n_atoms: int) -> list[tuple[int, int, float]]:
    bond_map: dict[tuple[int, int], float] = {}
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        cols = s.split()
        if len(cols) < 3:
            continue
        try:
            src = int(cols[0]) - 1
        except ValueError:
            continue
        if src < 0 or src >= n_atoms:
            continue
        tail = cols[1:]
        if len(tail) % 2 != 0:
            continue
        ok = True
        parsed: list[tuple[int, float]] = []
        for i in range(0, len(tail), 2):
            try:
                dst = int(tail[i]) - 1
                order = float(tail[i + 1])
            except ValueError:
                ok = False
                break
            if dst < 0 or dst >= n_atoms or dst == src:
                continue
            parsed.append((int(dst), float(order)))
        if not ok:
            continue
        for dst, order in parsed:
            key = (src, dst) if src < dst else (dst, src)
            if key not in bond_map:
                bond_map[key] = float(order)
    return [(int(i), int(j), float(order)) for (i, j), order in sorted(bond_map.items())]
