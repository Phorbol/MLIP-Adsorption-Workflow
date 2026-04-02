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
    dedup_metric: str = "rmsd",
    signature_mode: str = "provenance",
    pose_overrides: dict[str, Any] | None = None,
    basin_overrides: dict[str, Any] | None = None,
) -> AdsorptionWorkflowConfig:
    basin_kwargs = {
        "relax_maxf": 0.1,
        "relax_steps": 80,
        "energy_window_ev": 2.5,
        "dedup_metric": str(dedup_metric),
        "signature_mode": str(signature_mode),
        "dedup_cluster_method": "hierarchical",
        "rmsd_threshold": 0.10,
        "desorption_min_bonds": 1,
        "work_dir": None,
    }
    if basin_overrides:
        basin_kwargs.update(dict(basin_overrides))
    return AdsorptionWorkflowConfig(
        work_dir=Path(work_dir),
        surface_preprocessor=SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=0.55),
            fallback_detector=VoxelFloodDetector(spacing=0.75),
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
