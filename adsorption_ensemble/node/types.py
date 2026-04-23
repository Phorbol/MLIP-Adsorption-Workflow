from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeConfig:
    bond_tau: float = 1.20
    node_hash_len: int = 20
    node_identity_mode: str = "legacy_absolute"


@dataclass
class ReactionNode:
    node_id: str
    node_id_legacy: str
    basin_id: int
    canonical_order: list[int]
    atomic_numbers: list[int]
    internal_bonds: list[tuple[int, int]]
    binding_pairs: list[tuple[int, int]]
    surface_env_key: str | None
    surface_geometry_key: str | None
    denticity: int
    relative_energy_ev: float | None
    provenance: dict
