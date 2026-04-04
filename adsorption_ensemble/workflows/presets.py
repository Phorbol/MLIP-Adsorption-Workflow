from __future__ import annotations

from pathlib import Path
from typing import Any

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveEmbeddingConfig
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from adsorption_ensemble.workflows.adsorption import AdsorptionWorkflowConfig


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
    dedup_metric: str = "binding_surface_distance",
    signature_mode: str = "provenance",
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
    mace_metric_aliases = {
        "mace",
        "mace_node_l2",
        "mace_l2",
        "pure_mace",
        "pure_mace_l2",
        "mace_only",
    }
    basin_kwargs = {
        "relax_maxf": 0.1,
        "relax_steps": 80,
        "energy_window_ev": 2.5,
        "dedup_metric": str(dedup_metric),
        "signature_mode": str(signature_mode),
        "dedup_cluster_method": ("greedy" if dedup_metric_norm in surface_metric_aliases else "hierarchical"),
        "rmsd_threshold": 0.10,
        "mace_node_l2_threshold": 0.20,
        "mace_device": "cuda",
        "mace_dtype": "float32",
        "final_basin_merge_metric": ("off" if dedup_metric_norm in mace_metric_aliases else "auto_mace"),
        "final_basin_merge_node_l2_threshold": 0.20,
        "final_basin_merge_cluster_method": "hierarchical",
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
        surface_preprocessor=SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=0.6),
            fallback_detector=VoxelFloodDetector(spacing=0.8),
            target_surface_fraction=None,
            target_count_mode="off",
        ),
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
