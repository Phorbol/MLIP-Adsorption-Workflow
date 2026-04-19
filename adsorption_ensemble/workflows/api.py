from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ase import Atoms

from adsorption_ensemble.selection import StageSelectionConfig
from adsorption_ensemble.workflows.adsorption import AdsorptionWorkflowResult, run_adsorption_workflow
from adsorption_ensemble.workflows.paper_readiness import PaperReadinessReport, evaluate_adsorption_workflow_readiness
from adsorption_ensemble.workflows.presets import (
    DEFAULT_BASIN_DEDUP_METRIC,
    DEFAULT_BASIN_SIGNATURE_MODE,
    make_adsorption_workflow_config,
)


@dataclass
class SamplingSchedule:
    name: str = "multistage"
    exhaustive_pose_sampling: bool = False
    run_conformer_search: bool = False
    pre_relax_selection: StageSelectionConfig = field(default_factory=StageSelectionConfig)
    post_relax_selection: StageSelectionConfig = field(default_factory=StageSelectionConfig)
    notes: str = (
        "Stage A: site-conditioned SE(3) oversampling; "
        "Stage B: geometric/clash pruning; "
        "Stage C: loose batch relax; "
        "Stage D: post-relax basin dedup."
    )


def list_sampling_schedule_presets() -> list[str]:
    return [
        "multistage_default",
        "multistage_iterative_fps_grid",
        "multistage_iterative_fps_site",
        "no_selection",
        "fps_then_molclus",
        "fps_then_hierarchical",
        "fps_then_fuzzy",
    ]


def make_sampling_schedule(
    preset: str = "multistage_default",
    *,
    exhaustive_pose_sampling: bool | None = None,
    run_conformer_search: bool | None = None,
    pre_relax_selection: StageSelectionConfig | None = None,
    post_relax_selection: StageSelectionConfig | None = None,
    notes: str | None = None,
) -> SamplingSchedule:
    key = str(preset).strip().lower()
    aliases = {
        "default": "multistage_default",
        "multistage": "multistage_default",
        "molclus_like_default": "multistage_default",
        "adaptive_default": "multistage_iterative_fps_grid",
        "iterative_fps_default": "multistage_iterative_fps_grid",
        "iterative_grid": "multistage_iterative_fps_grid",
        "iterative_site": "multistage_iterative_fps_site",
        "none": "no_selection",
        "off": "no_selection",
        "disabled": "no_selection",
        "fps_molclus": "fps_then_molclus",
        "fps_then_energy_rmsd_window": "fps_then_molclus",
        "fps_hierarchical": "fps_then_hierarchical",
        "fps_fuzzy": "fps_then_fuzzy",
    }
    canonical = aliases.get(key, key)
    if canonical == "multistage_default":
        schedule = SamplingSchedule(
            name="multistage_default",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="fps",
                max_candidates=24,
                descriptor="adsorbate_surface_distance",
                random_seed=0,
            ),
            post_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="energy_rmsd_window",
                energy_window_ev=3.0,
                rmsd_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            notes=(
                "Conservative multistage default: FPS trims the raw pose pool before relax; "
                "post-relax Molclus-like energy-window plus geometry diversity keeps low-energy "
                "coverage without assuming a fixed cluster count."
            ),
        )
    elif canonical == "multistage_iterative_fps_grid":
        schedule = SamplingSchedule(
            name="multistage_iterative_fps_grid",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="iterative_fps",
                max_candidates=24,
                descriptor="adsorbate_surface_distance",
                random_seed=0,
                fps_round_size=4,
                fps_rounds=12,
                grid_convergence=True,
                grid_convergence_pca_var=0.95,
                grid_convergence_grid_bins=12,
                grid_convergence_min_rounds=3,
                grid_convergence_patience=2,
                grid_convergence_min_coverage_gain=1e-3,
                grid_convergence_min_novelty=5e-2,
            ),
            post_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="energy_rmsd_window",
                energy_window_ev=3.0,
                rmsd_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            notes=(
                "Adaptive multistage preset: iterative FPS stops early when PCA-grid occupancy saturates; "
                "post-relax Molclus-like energy-window plus geometry diversity keeps low-energy coverage."
            ),
        )
    elif canonical == "multistage_iterative_fps_site":
        schedule = SamplingSchedule(
            name="multistage_iterative_fps_site",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="iterative_fps",
                max_candidates=24,
                descriptor="adsorbate_surface_distance",
                random_seed=0,
                fps_round_size=4,
                fps_rounds=12,
                occupancy_convergence=True,
                occupancy_min_new_bins=0,
                occupancy_patience=2,
                occupancy_min_rounds=2,
            ),
            post_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="energy_rmsd_window",
                energy_window_ev=3.0,
                rmsd_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            notes=(
                "Adaptive multistage preset: iterative FPS stops when new site/provenance bins plateau; "
                "post-relax Molclus-like filtering remains unchanged."
            ),
        )
    elif canonical == "no_selection":
        schedule = SamplingSchedule(
            name="no_selection",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(enabled=False, strategy="none"),
            post_relax_selection=StageSelectionConfig(enabled=False, strategy="none"),
            notes="Disable both pre-relax and post-relax subset selection for exhaustive ablations.",
        )
    elif canonical == "fps_then_molclus":
        schedule = SamplingSchedule(
            name="fps_then_molclus",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="fps",
                max_candidates=12,
                descriptor="adsorbate_surface_distance",
                random_seed=0,
            ),
            post_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="energy_rmsd_window",
                energy_window_ev=3.0,
                rmsd_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            notes="Aggressive prescreening schedule: FPS before relax, then Molclus-like post-relax filtering.",
        )
    elif canonical == "fps_then_hierarchical":
        schedule = SamplingSchedule(
            name="fps_then_hierarchical",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="fps",
                max_candidates=12,
                descriptor="adsorbate_surface_distance",
                random_seed=0,
            ),
            post_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="hierarchical",
                energy_window_ev=3.0,
                cluster_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            notes="FPS before relax, then threshold-based hierarchical representative selection after relax.",
        )
    elif canonical == "fps_then_fuzzy":
        schedule = SamplingSchedule(
            name="fps_then_fuzzy",
            exhaustive_pose_sampling=False,
            run_conformer_search=False,
            pre_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="fps",
                max_candidates=12,
                descriptor="adsorbate_surface_distance",
                random_seed=0,
            ),
            post_relax_selection=StageSelectionConfig(
                enabled=True,
                strategy="fuzzy",
                energy_window_ev=3.0,
                cluster_threshold=0.05,
                descriptor="adsorbate_surface_distance",
            ),
            notes="FPS before relax, then fuzzy cluster representative selection after relax.",
        )
    else:
        available = ", ".join(list_sampling_schedule_presets())
        raise ValueError(f"Unsupported sampling schedule preset: {preset}. Available presets: {available}")
    if exhaustive_pose_sampling is not None:
        schedule.exhaustive_pose_sampling = bool(exhaustive_pose_sampling)
    if run_conformer_search is not None:
        schedule.run_conformer_search = bool(run_conformer_search)
    if pre_relax_selection is not None:
        schedule.pre_relax_selection = StageSelectionConfig(**asdict(pre_relax_selection))
    if post_relax_selection is not None:
        schedule.post_relax_selection = StageSelectionConfig(**asdict(post_relax_selection))
    if notes is not None:
        schedule.notes = str(notes)
    return schedule


@dataclass
class AdsorptionEnsembleRequest:
    slab: Atoms
    adsorbate: Atoms
    work_dir: str | Path
    placement_mode: str = "anchor_free"
    schedule: SamplingSchedule = field(default_factory=SamplingSchedule)
    dedup_metric: str = DEFAULT_BASIN_DEDUP_METRIC
    signature_mode: str = DEFAULT_BASIN_SIGNATURE_MODE
    pose_overrides: dict[str, Any] = field(default_factory=dict)
    basin_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdsorptionEnsembleResult:
    request: AdsorptionEnsembleRequest
    workflow: AdsorptionWorkflowResult
    readiness: PaperReadinessReport
    files: dict[str, str]
    summary: dict[str, Any]


def generate_adsorption_ensemble(
    *,
    slab: Atoms,
    adsorbate: Atoms,
    work_dir: str | Path,
    placement_mode: str = "anchor_free",
    schedule: SamplingSchedule | None = None,
    dedup_metric: str = DEFAULT_BASIN_DEDUP_METRIC,
    signature_mode: str = DEFAULT_BASIN_SIGNATURE_MODE,
    pose_overrides: dict[str, Any] | None = None,
    basin_overrides: dict[str, Any] | None = None,
    basin_relax_backend: object | None = None,
    md_runner: object | None = None,
    conformer_descriptor_extractor: object | None = None,
    conformer_relax_backend: object | None = None,
) -> AdsorptionEnsembleResult:
    sched = schedule or SamplingSchedule()
    req = AdsorptionEnsembleRequest(
        slab=slab,
        adsorbate=adsorbate,
        work_dir=Path(work_dir),
        placement_mode=str(placement_mode),
        schedule=sched,
        dedup_metric=str(dedup_metric),
        signature_mode=str(signature_mode),
        pose_overrides=dict(pose_overrides or {}),
        basin_overrides=dict(basin_overrides or {}),
    )
    cfg = make_adsorption_workflow_config(
        work_dir=req.work_dir,
        placement_mode=req.placement_mode,
        single_atom=(len(adsorbate) == 1),
        exhaustive_pose_sampling=bool(req.schedule.exhaustive_pose_sampling),
        dedup_metric=req.dedup_metric,
        signature_mode=req.signature_mode,
        pose_overrides=req.pose_overrides,
        basin_overrides=req.basin_overrides,
    )
    cfg.run_conformer_search = bool(req.schedule.run_conformer_search)
    cfg.pre_relax_selection = req.schedule.pre_relax_selection
    cfg.basin_config.post_relax_selection = req.schedule.post_relax_selection
    workflow = run_adsorption_workflow(
        slab=slab,
        adsorbate=adsorbate,
        config=cfg,
        md_runner=md_runner,
        conformer_descriptor_extractor=conformer_descriptor_extractor,
        conformer_relax_backend=conformer_relax_backend,
        basin_relax_backend=basin_relax_backend,
    )
    readiness = evaluate_adsorption_workflow_readiness(workflow)
    files = dict(workflow.artifacts)
    files.setdefault("work_dir", Path(req.work_dir).as_posix())
    summary = {
        "placement_mode": str(req.placement_mode),
        "schedule": asdict(req.schedule),
        "dedup_metric": str(req.dedup_metric),
        "signature_mode": str(req.signature_mode),
        "n_surface_atoms": int(workflow.summary["n_surface_atoms"]),
        "n_basis_primitives": int(workflow.summary["n_basis_primitives"]),
        "n_pose_frames": int(workflow.summary["n_pose_frames"]),
        "n_pose_frames_selected_for_basin": int(workflow.summary.get("n_pose_frames_selected_for_basin", workflow.summary["n_pose_frames"])),
        "n_basins": int(workflow.summary["n_basins"]),
        "n_nodes": int(workflow.summary["n_nodes"]),
        "paper_readiness_score": int(readiness.score),
        "paper_readiness_max_score": int(readiness.max_score),
    }
    return AdsorptionEnsembleResult(
        request=req,
        workflow=workflow,
        readiness=readiness,
        files=files,
        summary=summary,
    )
