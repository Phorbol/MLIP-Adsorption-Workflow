from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate

from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import SurfacePreprocessor


def _atomic_number_features(slab: Atoms) -> np.ndarray:
    return np.asarray(slab.get_atomic_numbers(), dtype=float).reshape(-1, 1)


def _infer_site_coordination(site_name: str) -> int:
    lname = str(site_name).strip().lower()
    if lname in {"ontop", "top", "1c"}:
        return 1
    if "bridge" in lname or lname == "2c":
        return 2
    if lname in {"hollow", "fcc", "hcp", "3c"}:
        return 3
    if lname == "4c":
        return 4
    return 1


def manual_reference_height(adsorbate: Atoms, site_name: str) -> float:
    coord = _infer_site_coordination(site_name)
    if len(adsorbate) == 1:
        if coord >= 3:
            return 0.9
        if coord == 2:
            return 1.0
        return 1.1
    sampler = PoseSampler(PoseSamplerConfig())
    mol_class = sampler._classify_molecule_shape(adsorbate)
    base_floor = float(sampler._preferred_min_height(mol_class=mol_class, site_coordination=coord))
    if coord <= 1:
        return max(base_floor, 1.85)
    if coord == 2:
        return max(base_floor, 1.70)
    if coord == 3:
        return max(base_floor, 1.55)
    return max(base_floor, 1.45)


def _make_frame(
    slab: Atoms,
    reference_ads: Atoms,
    *,
    binding_index: int,
    site_name: str,
    site_label: str,
    reference_source: str,
    site_kind: str | None,
    primitive_index: int | None,
    basis_id: int | None,
    primitive_atom_ids: tuple[int, ...] | None,
    site_center: np.ndarray | None,
    site_normal: np.ndarray | None,
) -> tuple[Atoms, dict[str, Any]]:
    frame = slab.copy()
    frame += reference_ads.copy()
    frame.info["generator"] = "reference_frame"
    frame.info["site_name"] = str(site_name)
    frame.info["site_label"] = str(site_label)
    frame.info["reference_source"] = str(reference_source)
    frame.info["mol_index"] = int(binding_index)
    frame.info["binding_atom_index"] = int(binding_index)
    frame.info["binding_atom_symbol"] = str(reference_ads[int(binding_index)].symbol)
    if site_kind is not None:
        frame.info["site_kind"] = str(site_kind)
    if primitive_index is not None:
        frame.info["primitive_index"] = int(primitive_index)
    if basis_id is not None:
        frame.info["basis_id"] = int(basis_id)
    if primitive_atom_ids is not None:
        frame.info["primitive_atom_ids"] = [int(i) for i in primitive_atom_ids]
    meta: dict[str, Any] = {
        "site_name": str(site_name),
        "site_label": str(site_label),
        "reference_source": str(reference_source),
        "mol_index": int(binding_index),
        "binding_atom_index": int(binding_index),
        "binding_atom_symbol": str(reference_ads[int(binding_index)].symbol),
    }
    if site_kind is not None:
        meta["site_kind"] = str(site_kind)
    if primitive_index is not None:
        meta["primitive_index"] = int(primitive_index)
    if basis_id is not None:
        meta["basis_id"] = int(basis_id)
    if primitive_atom_ids is not None:
        meta["primitive_atom_ids"] = [int(i) for i in primitive_atom_ids]
    if site_center is not None:
        meta["site_center"] = [float(x) for x in np.asarray(site_center, dtype=float).reshape(3)]
    if site_normal is not None:
        meta["site_normal"] = [float(x) for x in np.asarray(site_normal, dtype=float).reshape(3)]
    return frame, meta


def _build_primitive_basis_reference_frames(
    slab: Atoms,
    adsorbate: Atoms,
    *,
    binding_index: int,
    reference_ads: Atoms,
) -> tuple[list[Atoms], list[dict[str, Any]]]:
    ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
    raw_primitives = PrimitiveBuilder().build(slab, ctx)
    embed_result = PrimitiveEmbedder(PrimitiveEmbeddingConfig()).fit_transform(
        slab=slab,
        primitives=list(raw_primitives),
        atom_features=_atomic_number_features(slab),
    )
    frames: list[Atoms] = []
    meta: list[dict[str, Any]] = []
    for primitive_index, primitive in enumerate(embed_result.basis_primitives):
        height = manual_reference_height(adsorbate, str(getattr(primitive, "site_label", None) or primitive.kind))
        placed_ads = reference_ads.copy()
        translation = np.asarray(primitive.center, dtype=float) + float(height) * np.asarray(primitive.normal, dtype=float)
        placed_ads.positions = np.asarray(placed_ads.positions, dtype=float) + translation.reshape(1, 3)
        basis_id = None if primitive.basis_id is None else int(primitive.basis_id)
        site_name = (
            str(primitive.site_label)
            if getattr(primitive, "site_label", None) is not None
            else f"{primitive.kind}|basis={basis_id if basis_id is not None else primitive_index}"
        )
        frame, row = _make_frame(
            slab,
            placed_ads,
            binding_index=int(binding_index),
            site_name=site_name,
            site_label=site_name,
            reference_source="primitive_basis_fallback",
            site_kind=str(primitive.kind),
            primitive_index=int(primitive_index),
            basis_id=basis_id,
            primitive_atom_ids=tuple(int(i) for i in primitive.atom_ids),
            site_center=np.asarray(primitive.center, dtype=float),
            site_normal=np.asarray(primitive.normal, dtype=float),
        )
        frames.append(frame)
        meta.append(row)
    return frames, meta


def build_ase_reference_frames(slab: Atoms, adsorbate: Atoms) -> tuple[list[Atoms], list[dict[str, Any]]]:
    info = slab.info.get("adsorbate_info", {})
    sites = info.get("sites", {}) if isinstance(info, dict) else {}
    binding_index = int(PoseSampler._select_binding_atom_index(adsorbate))
    reference_ads = PoseSampler.orient_adsorbate_for_binding(
        adsorbate,
        binding_atom_index=binding_index,
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    frames: list[Atoms] = []
    meta: list[dict[str, Any]] = []
    for site_name in sites.keys():
        placed = slab.copy()
        add_adsorbate(
            placed,
            reference_ads.copy(),
            manual_reference_height(adsorbate, str(site_name)),
            str(site_name),
            mol_index=binding_index,
        )
        frame, row = _make_frame(
            slab=slab,
            reference_ads=placed[len(slab) :],
            binding_index=int(binding_index),
            site_name=str(site_name),
            site_label=str(site_name).lower(),
            reference_source="ase_adsorbate_info",
            site_kind=None,
            primitive_index=None,
            basis_id=None,
            primitive_atom_ids=None,
            site_center=None,
            site_normal=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        frames.append(frame)
        meta.append(row)
    if frames:
        return frames, meta
    return _build_primitive_basis_reference_frames(
        slab=slab,
        adsorbate=adsorbate,
        binding_index=int(binding_index),
        reference_ads=reference_ads,
    )
