from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.build import molecule


def make_glucose_chain_like() -> Atoms:
    symbols = []
    positions = []
    for i in range(6):
        x = 1.45 * i
        symbols.append("C")
        positions.append((x, 0.0, 0.12 * ((-1) ** i)))
        symbols.append("O")
        positions.append((x + 0.45, 1.10, 0.25 * ((-1) ** (i + 1))))
    for i in range(12):
        x = 0.75 * i
        y = -1.0 if i % 2 == 0 else -1.55
        z = 0.35 * ((-1) ** i)
        symbols.append("H")
        positions.append((x, y, z))
    return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))


def make_glucose_ring_like() -> Atoms:
    ring = []
    symbols = []
    radius = 1.55
    for i in range(5):
        ang = 2.0 * math.pi * i / 5.0
        ring.append((radius * math.cos(ang), radius * math.sin(ang), 0.1 * ((-1) ** i)))
        symbols.append("C")
    ring.append((0.0, -radius * 0.2, 0.35))
    symbols.append("O")
    positions = list(ring)
    for i, (x, y, z) in enumerate(ring):
        ang = math.atan2(y, x)
        positions.append((x + 0.95 * math.cos(ang), y + 0.95 * math.sin(ang), z + 0.5))
        symbols.append("O")
    for i in range(12):
        ang = 2.0 * math.pi * i / 12.0
        positions.append((2.8 * math.cos(ang), 2.8 * math.sin(ang), 0.7 * ((-1) ** i)))
        symbols.append("H")
    return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))


def make_glycine_like() -> Atoms:
    symbols = ["N", "C", "C", "O", "O", "H", "H", "H", "H", "H"]
    positions = np.asarray(
        [
            (-1.25, 0.05, 0.10),
            (0.00, 0.00, 0.00),
            (1.45, 0.00, 0.05),
            (2.15, 1.00, -0.10),
            (2.10, -1.05, 0.18),
            (-1.65, -0.80, 0.30),
            (-1.70, 0.82, -0.20),
            (0.05, 1.02, -0.28),
            (0.02, -0.95, 0.45),
            (2.95, -1.10, -0.25),
        ],
        dtype=float,
    )
    return Atoms(symbols=symbols, positions=positions)


def make_dipeptide_like() -> Atoms:
    base = make_glycine_like()
    ext_symbols = ["N", "C", "C", "O", "H", "H", "H"]
    ext_positions = np.asarray(
        [
            (3.25, 0.10, 0.00),
            (4.55, 0.08, 0.10),
            (5.80, -0.05, -0.15),
            (6.55, 0.85, -0.30),
            (3.00, 0.95, 0.35),
            (4.65, 0.95, 0.45),
            (5.90, -0.95, 0.20),
        ],
        dtype=float,
    )
    merged = base + Atoms(symbols=ext_symbols, positions=ext_positions)
    return merged


def make_para_nitrochlorobenzene_like() -> Atoms:
    ring = molecule("C6H6")
    positions = np.asarray(ring.positions, dtype=float)
    symbols = ring.get_chemical_symbols()
    max_x = int(np.argmax(positions[:, 0]))
    min_x = int(np.argmin(positions[:, 0]))
    positions[max_x] += np.asarray([1.45, 0.00, 0.00], dtype=float)
    symbols[max_x] = "Cl"
    extra = Atoms(
        symbols=["N", "O", "O"],
        positions=np.asarray(
            [
                positions[min_x] + np.asarray([-1.35, 0.00, 0.00], dtype=float),
                positions[min_x] + np.asarray([-2.35, 0.65, 0.10], dtype=float),
                positions[min_x] + np.asarray([-2.35, -0.65, -0.10], dtype=float),
            ],
            dtype=float,
        ),
    )
    out = Atoms(symbols=symbols, positions=positions) + extra
    return out


def make_para_nitrobenzoic_acid_like() -> Atoms:
    ring = molecule("C6H6")
    positions = np.asarray(ring.positions, dtype=float)
    symbols = ring.get_chemical_symbols()
    min_x = int(np.argmin(positions[:, 0]))
    max_x = int(np.argmax(positions[:, 0]))
    extra = Atoms(
        symbols=["N", "O", "O", "C", "O", "O", "H"],
        positions=np.asarray(
            [
                positions[min_x] + np.asarray([-1.35, 0.00, 0.00], dtype=float),
                positions[min_x] + np.asarray([-2.35, 0.65, 0.10], dtype=float),
                positions[min_x] + np.asarray([-2.35, -0.65, -0.10], dtype=float),
                positions[max_x] + np.asarray([1.45, 0.00, 0.00], dtype=float),
                positions[max_x] + np.asarray([2.55, 0.40, 0.12], dtype=float),
                positions[max_x] + np.asarray([2.45, -0.85, -0.10], dtype=float),
                positions[max_x] + np.asarray([3.15, -1.35, -0.25], dtype=float),
            ],
            dtype=float,
        ),
    )
    return Atoms(symbols=symbols, positions=positions) + extra


def get_test_adsorbate_cases() -> dict[str, Atoms]:
    return {
        "CO": molecule("CO"),
        "H2O": molecule("H2O"),
        "NH3": molecule("NH3"),
        "CH3OH": molecule("CH3OH"),
        "C2H6": molecule("C2H6"),
        "C2H4": molecule("C2H4"),
        "C2H2": molecule("C2H2"),
        "C6H6": molecule("C6H6"),
        "glucose_chain_like": make_glucose_chain_like(),
        "glucose_ring_like": make_glucose_ring_like(),
        "glycine_like": make_glycine_like(),
        "dipeptide_like": make_dipeptide_like(),
        "p_nitrochlorobenzene_like": make_para_nitrochlorobenzene_like(),
        "p_nitrobenzoic_acid_like": make_para_nitrobenzoic_acid_like(),
    }
