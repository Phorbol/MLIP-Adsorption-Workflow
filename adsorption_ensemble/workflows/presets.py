from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from adsorption_ensemble.workflows.adsorption import AdsorptionWorkflowConfig

DEFAULT_BASIN_DEDUP_METRIC = "mace_node_l2"
DEFAULT_BASIN_SIGNATURE_MODE = "provenance"
DEFAULT_BASIN_CLUSTER_METHOD = "hierarchical"
DEFAULT_MACE_NODE_L2_THRESHOLD = 0.20
DEFAULT_FINAL_BASIN_MERGE_METRIC = "auto_ref_canonical_mace"
DEFAULT_FINAL_BASIN_MERGE_NODE_L2_THRESHOLD = 0.02
DEFAULT_MACE_MODEL_PATH = str(os.environ.get("AE_MACE_MODEL_PATH", "/root/.cache/mace/mace-omat-0-small.model")).strip()
DEFAULT_MACE_HEAD_NAME = "omat_pbe"
DEFAULT_SURFACE_TARGET_MODE = "adaptive"
DEFAULT_SURFACE_TARGET_FRACTION = 0.25


def make_default_surface_preprocessor(
    *,
    grid_step: float = 0.6,
    spacing: float = 0.8,
    target_count_mode: str = DEFAULT_SURFACE_TARGET_MODE,
    target_surface_fraction: float | None = DEFAULT_SURFACE_TARGET_FRACTION,
    overrides: dict[str, Any] | None = None,
) -> SurfacePreprocessor:
    kwargs: dict[str, Any] = {
        "min_surface_atoms": 6,
        "primary_detector": ProbeScanDetector(grid_step=float(grid_step)),
        "fallback_detector": VoxelFloodDetector(spacing=float(spacing)),
        "target_surface_fraction": target_surface_fraction,
        "target_count_mode": str(target_count_mode),
    }
    if overrides:
        kwargs.update(dict(overrides))
    return SurfacePreprocessor(**kwargs)


def make_pose_sampler_config(
    *,
    placement_mode: str = "anchor_free",
    single_atom: bool = False,
    exhaustive: bool = False,
    overrides: dict[str, Any] | None = None,
) -> PoseSamplerConfig:
    base: dict[str, Any]
    if single_atom:
        base = {
            "placement_mode": str(placement_mode),
            "n_rotations": 1,
            "n_azimuth": 1,
            "n_shifts": 1,
            "shift_radius": 0.0,
            "min_height": 0.8,
            "max_height": 2.4,
            "height_step": 0.10,
            "max_poses_per_site": (6 if exhaustive else 3),
            "random_seed": 0,
            "adaptive_height_fallback": True,
            "adaptive_height_fallback_step": 0.20,
            "adaptive_height_fallback_max_extra": 1.60,
            "adaptive_height_fallback_contact_slack": 0.60,
        }
    else:
        base = {
            "placement_mode": str(placement_mode),
            "n_rotations": (8 if exhaustive else 4),
            "n_azimuth": (12 if exhaustive else 8),
            "n_shifts": (3 if exhaustive else 2),
            "shift_radius": (0.25 if exhaustive else 0.15),
            "min_height": 1.2,
            "max_height": (3.8 if exhaustive else 3.4),
            "height_step": 0.10,
            "max_poses_per_site": (12 if exhaustive else 4),
            "random_seed": 0,
            "adaptive_height_fallback": True,
            "adaptive_height_fallback_step": 0.20,
            "adaptive_height_fallback_max_extra": 1.60,
            "adaptive_height_fallback_contact_slack": 0.60,
        }
    if overrides:
        base.update(dict(overrides))
    return PoseSamplerConfig(**base)


def make_adsorption_workflow_config(
    work_dir: str | Path,
    *,
    placement_mode: str = "anchor_free",
    single_atom: bool = False,
    exhaustive_pose_sampling: bool = False,
    dedup_metric: str = DEFAULT_BASIN_DEDUP_METRIC,
    signature_mode: str = DEFAULT_BASIN_SIGNATURE_MODE,
    pose_overrides: dict[str, Any] | None = None,
    basin_overrides: dict[str, Any] | None = None,
) -> AdsorptionWorkflowConfig:
    dedup_metric_norm = str(dedup_metric).strip().lower()
    surface_metric_aliases = {
        "binding_surface",
        "binding_surface_distance",
        "surface_binding_distance",
        "binding_surface_descriptor",
    }
    basin_kwargs = {
        "relax_maxf": 0.1,
        "relax_steps": 80,
        "energy_window_ev": 2.5,
        "dedup_metric": str(dedup_metric),
        "signature_mode": str(signature_mode),
        "dedup_cluster_method": DEFAULT_BASIN_CLUSTER_METHOD,
        "rmsd_threshold": 0.10,
        "mace_node_l2_threshold": DEFAULT_MACE_NODE_L2_THRESHOLD,
        "mace_model_path": (DEFAULT_MACE_MODEL_PATH or None),
        "mace_device": "cuda",
        "mace_dtype": "float64",
        "mace_head_name": DEFAULT_MACE_HEAD_NAME,
        "final_basin_merge_metric": DEFAULT_FINAL_BASIN_MERGE_METRIC,
        "final_basin_merge_node_l2_threshold": DEFAULT_FINAL_BASIN_MERGE_NODE_L2_THRESHOLD,
        "final_basin_merge_cluster_method": DEFAULT_BASIN_CLUSTER_METHOD,
        "desorption_min_bonds": 1,
        "work_dir": None,
    }
    if dedup_metric_norm in surface_metric_aliases:
        basin_kwargs.update(
            {
                "surface_descriptor_threshold": 0.30,
                "surface_descriptor_nearest_k": 8,
                "surface_descriptor_atom_mode": "binding_only",
                "surface_descriptor_relative": False,
                "surface_descriptor_rmsd_gate": 0.25,
            }
        )
    if basin_overrides:
        basin_kwargs.update(dict(basin_overrides))
    return AdsorptionWorkflowConfig(
        work_dir=Path(work_dir),
        surface_preprocessor=make_default_surface_preprocessor(),
        pose_sampler_config=make_pose_sampler_config(
            placement_mode=str(placement_mode),
            single_atom=bool(single_atom),
            exhaustive=bool(exhaustive_pose_sampling),
            overrides=pose_overrides,
        ),
        basin_config=BasinConfig(**basin_kwargs),
        max_selected_primitives=24,
        save_basin_dictionary=True,
        save_site_visualizations=True,
        save_raw_site_dictionary=True,
        save_selected_site_dictionary=True,
        primitive_embedding_config=PrimitiveEmbeddingConfig(l2_distance_threshold=0.20),
    )
