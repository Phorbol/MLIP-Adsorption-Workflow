from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms


@dataclass
class BasinConfig:
    relax_maxf: float = 0.10
    relax_steps: int = 80
    energy_window_ev: float = 0.20
    dedup_metric: str = "rmsd"
    rmsd_threshold: float = 0.10
    mace_node_l2_threshold: float = 2.0
    mace_model_path: str | None = None
    mace_device: str = "cpu"
    mace_dtype: str = "float32"
    mace_max_edges_per_batch: int = 15000
    mace_layers_to_keep: int = -1
    mace_head_name: str | None = None
    mace_mlp_energy_key: str | None = None
    binding_tau: float = 1.15
    desorption_min_bonds: int = 1
    surface_reconstruction_max_disp: float = 0.50
    dissociation_allow_bond_change: bool = False
    burial_margin: float = 0.30
    work_dir: Path | None = None


@dataclass
class RejectedCandidate:
    candidate_id: int
    reason: str
    metrics: dict


@dataclass
class Basin:
    basin_id: int
    atoms: Atoms
    energy_ev: float
    member_candidate_ids: list[int]
    binding_pairs: list[tuple[int, int]]
    denticity: int
    signature: str


@dataclass
class BasinResult:
    basins: list[Basin]
    rejected: list[RejectedCandidate]
    relax_backend: str
    summary: dict
