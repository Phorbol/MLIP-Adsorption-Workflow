from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from adsorption_ensemble.basin import BasinBuilder, BasinConfig, BasinResult, build_basin_dictionary, run_basin_ablation
from adsorption_ensemble.conformer_md import ConformerEnsemble, ConformerMDSampler, ConformerMDSamplerConfig
from adsorption_ensemble.node import NodeConfig, ReactionNode, basin_to_node
from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.selection import StageSelectionConfig, apply_stage_selection, stage_selection_summary
from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig, build_site_dictionary
from adsorption_ensemble.surface import SurfaceContext, SurfacePreprocessor, export_surface_detection_report
from adsorption_ensemble.visualization import (
    plot_inequivalent_sites_2d,
    plot_site_centers_only,
    plot_site_embedding_pca,
    plot_surface_primitives_2d,
)
from adsorption_ensemble.basin.reporting import describe_basin_binding, summarize_basin_member_provenance


@dataclass
class AdsorptionWorkflowConfig:
    work_dir: Path = field(default_factory=lambda: Path("artifacts") / "adsorption_workflow")
    surface_preprocessor: SurfacePreprocessor = field(default_factory=lambda: SurfacePreprocessor(min_surface_atoms=6))
    primitive_builder: PrimitiveBuilder = field(default_factory=PrimitiveBuilder)
    pose_sampler_config: PoseSamplerConfig = field(default_factory=PoseSamplerConfig)
    basin_config: BasinConfig = field(default_factory=BasinConfig)
    node_config: NodeConfig = field(default_factory=NodeConfig)
    run_conformer_search: bool = False
    conformer_config: ConformerMDSamplerConfig = field(default_factory=ConformerMDSamplerConfig)
    conformer_job_name: str = "conformer_search"
    max_primitives: int | None = None
    max_selected_primitives: int | None = None
    save_surface_report: bool = True
    save_site_dictionary: bool = True
    save_pose_pool: bool = True
    save_basin_dictionary: bool = True
    save_basin_ablation: bool = False
    basin_ablation_metrics: tuple[str, ...] = ("signature_only", "rmsd")
    save_site_visualizations: bool = True
    save_raw_site_dictionary: bool = True
    save_selected_site_dictionary: bool = True
    primitive_embedding_config: PrimitiveEmbeddingConfig = field(default_factory=PrimitiveEmbeddingConfig)
    pre_relax_selection: StageSelectionConfig = field(default_factory=StageSelectionConfig)


@dataclass
class AdsorptionWorkflowResult:
    surface_context: SurfaceContext
    primitives: list[Any]
    conformers: list[Atoms]
    pose_frames: list[Atoms]
    basin_result: BasinResult
    nodes: list[ReactionNode]
    artifacts: dict[str, str]
    summary: dict[str, Any]
    conformer_result: ConformerEnsemble | None = None


def run_adsorption_workflow(
    slab: Atoms,
    adsorbate: Atoms,
    config: AdsorptionWorkflowConfig | None = None,
    *,
    md_runner: object | None = None,
    conformer_descriptor_extractor: object | None = None,
    conformer_relax_backend: object | None = None,
    basin_relax_backend: object | None = None,
) -> AdsorptionWorkflowResult:
    cfg = config or AdsorptionWorkflowConfig()
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ctx = cfg.surface_preprocessor.build_context(slab)
    surface_report_dir = work_dir / "surface_report"
    if bool(cfg.save_surface_report):
        export_surface_detection_report(slab, ctx, surface_report_dir)

    raw_primitives = cfg.primitive_builder.build(slab, ctx)
    if cfg.max_primitives is not None:
        raw_primitives = raw_primitives[: max(0, int(cfg.max_primitives))]
    primitives = list(raw_primitives)
    atom_features = _make_atomic_number_features(slab)
    embed_result = PrimitiveEmbedder(cfg.primitive_embedding_config).fit_transform(slab=slab, primitives=primitives, atom_features=atom_features)
    # Sampling should operate on inequivalent-site representatives.
    primitives = list(embed_result.basis_primitives)
    if cfg.max_selected_primitives is not None:
        primitives = primitives[: max(0, int(cfg.max_selected_primitives))]

    artifacts: dict[str, str] = {}
    if bool(cfg.save_raw_site_dictionary):
        raw_site_dictionary = build_site_dictionary(embed_result.primitives)
        raw_site_dictionary_path = work_dir / "raw_site_dictionary.json"
        _write_json(raw_site_dictionary_path, raw_site_dictionary)
        artifacts["raw_site_dictionary_json"] = raw_site_dictionary_path.as_posix()
    if bool(cfg.save_selected_site_dictionary):
        selected_site_dictionary = build_site_dictionary(primitives)
        selected_site_dictionary_path = work_dir / "selected_site_dictionary.json"
        _write_json(selected_site_dictionary_path, selected_site_dictionary)
        artifacts["selected_site_dictionary_json"] = selected_site_dictionary_path.as_posix()
        if bool(cfg.save_site_dictionary):
            site_dictionary_path = work_dir / "site_dictionary.json"
            _write_json(site_dictionary_path, selected_site_dictionary)
            artifacts["site_dictionary_json"] = site_dictionary_path.as_posix()
    if bool(cfg.save_site_visualizations):
        sites_png = work_dir / "sites.png"
        sites_only_png = work_dir / "sites_only.png"
        sites_ineq_png = work_dir / "sites_inequivalent.png"
        plot_surface_primitives_2d(slab=slab, context=ctx, primitives=embed_result.primitives, filename=sites_png)
        plot_site_centers_only(slab=slab, primitives=embed_result.primitives, filename=sites_only_png)
        plot_inequivalent_sites_2d(slab=slab, primitives=embed_result.primitives, filename=sites_ineq_png)
        artifacts["sites_png"] = sites_png.as_posix()
        artifacts["sites_only_png"] = sites_only_png.as_posix()
        artifacts["sites_inequivalent_png"] = sites_ineq_png.as_posix()
        site_pca_png = work_dir / "site_embedding_pca.png"
        plot_site_embedding_pca(embed_result.primitives, filename=site_pca_png)
        artifacts["site_embedding_pca_png"] = site_pca_png.as_posix()

    conformer_result: ConformerEnsemble | None = None
    conformers = [adsorbate.copy()]
    if bool(cfg.run_conformer_search):
        conformer_sampler = ConformerMDSampler(
            config=cfg.conformer_config,
            md_runner=md_runner,
            descriptor_extractor=conformer_descriptor_extractor,
            relax_backend=conformer_relax_backend,
        )
        conformer_result = conformer_sampler.run(adsorbate.copy(), job_name=str(cfg.conformer_job_name))
        conformers = [a.copy() for a in conformer_result.conformers] or [adsorbate.copy()]
        conformer_meta_path = work_dir / "conformer_metadata.json"
        _write_json(conformer_meta_path, dict(conformer_result.metadata))
        artifacts["conformer_metadata_json"] = conformer_meta_path.as_posix()

    pose_sampler = PoseSampler(cfg.pose_sampler_config)
    pose_frames: list[Atoms] = []
    for conformer_id, conformer in enumerate(conformers):
        poses = _sample_with_fallback(
            sampler=pose_sampler,
            slab=slab,
            adsorbate=conformer,
            primitives=primitives,
            surface_atom_ids=ctx.detection.surface_atom_ids,
        )
        for pose in poses:
            frame = slab + pose.atoms
            primitive = primitives[int(pose.primitive_index)]
            frame.info["conformer_id"] = int(conformer_id)
            frame.info["primitive_index"] = int(pose.primitive_index)
            frame.info["basis_id"] = (-1 if pose.basis_id is None else int(pose.basis_id))
            frame.info["site_kind"] = str(primitive.kind)
            frame.info["site_label"] = (
                str(primitive.site_label)
                if getattr(primitive, "site_label", None) is not None
                else f"{primitive.kind}|basis={(-1 if pose.basis_id is None else int(pose.basis_id))}"
            )
            frame.info["rotation_index"] = int(pose.rotation_index)
            frame.info["azimuth_index"] = int(pose.azimuth_index)
            frame.info["height_shift_index"] = int(pose.height_shift_index)
            frame.info["height"] = float(pose.height)
            frame.info["placement_mode"] = str(cfg.pose_sampler_config.placement_mode)
            frame.info["anchor_free_reference"] = str(cfg.pose_sampler_config.anchor_free_reference)
            pose_frames.append(frame)
    if bool(cfg.save_pose_pool) and pose_frames:
        pose_pool_path = work_dir / "pose_pool.extxyz"
        write(pose_pool_path.as_posix(), pose_frames)
        artifacts["pose_pool_extxyz"] = pose_pool_path.as_posix()

    pre_relax_selected_ids, pre_relax_diag = apply_stage_selection(
        frames=pose_frames,
        config=cfg.pre_relax_selection,
        slab_n=len(slab),
        energies=None,
    )
    basin_input_frames = [pose_frames[i] for i in pre_relax_selected_ids]
    if basin_input_frames and bool(cfg.pre_relax_selection.enabled):
        basin_input_path = work_dir / "pose_pool_selected.extxyz"
        write(basin_input_path.as_posix(), basin_input_frames)
        artifacts["pose_pool_selected_extxyz"] = basin_input_path.as_posix()
        pre_relax_diag_path = work_dir / "pre_relax_selection.json"
        _write_json(pre_relax_diag_path, pre_relax_diag)
        artifacts["pre_relax_selection_json"] = pre_relax_diag_path.as_posix()

    basin_cfg = cfg.basin_config
    basin_cfg.work_dir = work_dir / "basin_work"
    basin_result = BasinBuilder(config=basin_cfg, relax_backend=basin_relax_backend).build(
        frames=basin_input_frames,
        slab_ref=slab,
        adsorbate_ref=adsorbate,
        slab_n=len(slab),
        normal_axis=int(ctx.classification.normal_axis),
    )

    basin_frames = []
    for basin in basin_result.basins:
        a = basin.atoms.copy()
        a.info["basin_id"] = int(basin.basin_id)
        a.info["energy_ev"] = float(basin.energy_ev)
        a.info["signature"] = str(basin.signature)
        basin_frames.append(a)
    if basin_frames:
        basins_extxyz = work_dir / "basins.extxyz"
        write(basins_extxyz.as_posix(), basin_frames)
        artifacts["basins_extxyz"] = basins_extxyz.as_posix()
    basins_json = work_dir / "basins.json"
    _write_json(
        basins_json,
        {
            "summary": dict(basin_result.summary),
            "relax_backend": str(basin_result.relax_backend),
            "basins": [
                {
                    "basin_id": int(b.basin_id),
                    "energy_ev": float(b.energy_ev),
                    "denticity": int(b.denticity),
                    "signature": str(b.signature),
                    "member_candidate_ids": [int(x) for x in b.member_candidate_ids],
                    "binding_pairs": [(int(i), int(j)) for i, j in b.binding_pairs],
                    **describe_basin_binding(basin_atoms=b.atoms, slab_n=len(slab), binding_pairs=list(b.binding_pairs)),
                    **summarize_basin_member_provenance(
                        [pose_frames[int(x)] for x in b.member_candidate_ids if 0 <= int(x) < len(pose_frames)]
                    ),
                }
                for b in basin_result.basins
            ],
            "rejected": [
                {"candidate_id": int(r.candidate_id), "reason": str(r.reason), "metrics": dict(r.metrics)}
                for r in basin_result.rejected
            ],
        },
    )
    artifacts["basins_json"] = basins_json.as_posix()

    energy_min = basin_result.summary.get("energy_min_ev", None)
    try:
        energy_min_ev = None if energy_min is None else float(energy_min)
    except Exception:
        energy_min_ev = None
    nodes = [basin_to_node(b, slab_n=len(slab), cfg=cfg.node_config, energy_min_ev=energy_min_ev) for b in basin_result.basins]
    nodes_json = work_dir / "nodes.json"
    _write_json(
        nodes_json,
        [
            {
                "node_id": str(n.node_id),
                "basin_id": int(n.basin_id),
                "canonical_order": [int(x) for x in n.canonical_order],
                "atomic_numbers": [int(x) for x in n.atomic_numbers],
                "internal_bonds": [(int(i), int(j)) for i, j in n.internal_bonds],
                "binding_pairs": [(int(i), int(j)) for i, j in n.binding_pairs],
                "denticity": int(n.denticity),
                "relative_energy_ev": (None if n.relative_energy_ev is None else float(n.relative_energy_ev)),
                "provenance": dict(n.provenance),
            }
            for n in nodes
        ],
    )
    artifacts["nodes_json"] = nodes_json.as_posix()

    if bool(cfg.save_basin_dictionary):
        basin_dict = build_basin_dictionary(
            basin_result,
            pose_frames=pose_frames,
            nodes=nodes,
            slab_n=len(slab),
        )
        basin_dict_path = work_dir / "basin_dictionary.json"
        _write_json(basin_dict_path, basin_dict)
        artifacts["basin_dictionary_json"] = basin_dict_path.as_posix()
    if bool(cfg.save_basin_ablation):
        basin_ablation = run_basin_ablation(
            frames=pose_frames,
            slab_ref=slab,
            adsorbate_ref=adsorbate,
            slab_n=len(slab),
            normal_axis=int(ctx.classification.normal_axis),
            base_config=cfg.basin_config,
            relax_backend=basin_relax_backend,
            metrics=tuple(cfg.basin_ablation_metrics),
        )
        basin_ablation_path = work_dir / "basin_ablation.json"
        _write_json(basin_ablation_path, basin_ablation)
        artifacts["basin_ablation_json"] = basin_ablation_path.as_posix()

    summary = {
        "n_surface_atoms": int(len(ctx.detection.surface_atom_ids)),
        "n_primitives": int(len(primitives)),
        "n_raw_primitives": int(len(embed_result.primitives)),
        "n_basis_primitives": int(len(embed_result.basis_primitives)),
        "n_conformers": int(len(conformers)),
        "n_pose_frames": int(len(pose_frames)),
        "n_pose_frames_selected_for_basin": int(len(basin_input_frames)),
        "n_basins": int(len(basin_result.basins)),
        "n_nodes": int(len(nodes)),
        "run_conformer_search": bool(cfg.run_conformer_search),
        "pre_relax_selection": stage_selection_summary(cfg.pre_relax_selection),
    }
    summary_path = work_dir / "workflow_summary.json"
    _write_json(summary_path, summary)
    artifacts["workflow_summary_json"] = summary_path.as_posix()
    if bool(cfg.save_surface_report):
        artifacts["surface_report_dir"] = surface_report_dir.as_posix()

    return AdsorptionWorkflowResult(
        surface_context=ctx,
        primitives=primitives,
        conformers=conformers,
        pose_frames=pose_frames,
        basin_result=basin_result,
        nodes=nodes,
        artifacts=artifacts,
        summary=summary,
        conformer_result=conformer_result,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sample_with_fallback(
    sampler: PoseSampler,
    slab: Atoms,
    adsorbate: Atoms,
    primitives: list[Any],
    surface_atom_ids: list[int],
) -> list[Any]:
    poses = sampler.sample(
        slab=slab,
        adsorbate=adsorbate,
        primitives=primitives,
        surface_atom_ids=surface_atom_ids,
    )
    if poses:
        return poses
    span = _adsorbate_span(adsorbate)
    retry_cfg = PoseSamplerConfig(**vars(sampler.config))
    retry_cfg.min_height = float(max(retry_cfg.min_height, 1.8))
    retry_cfg.max_height = float(max(retry_cfg.max_height + 0.8, retry_cfg.min_height + 1.0 + 0.15 * span))
    retry_cfg.height_step = float(min(retry_cfg.height_step, 0.15))
    retry_cfg.clash_tau = float(max(0.65, retry_cfg.clash_tau - 0.1))
    retry_cfg.site_contact_tolerance = float(retry_cfg.site_contact_tolerance + 0.15)
    retry_cfg.random_seed = int(retry_cfg.random_seed) + 17
    retry_sampler = PoseSampler(retry_cfg)
    poses = retry_sampler.sample(
        slab=slab,
        adsorbate=adsorbate,
        primitives=primitives,
        surface_atom_ids=surface_atom_ids,
    )
    if poses:
        return poses
    # Second-stage fallback for hard surfaces (e.g., covalent/oxide terminations):
    # start from a higher height window and looser contact constraints.
    retry2_cfg = PoseSamplerConfig(**vars(retry_cfg))
    retry2_cfg.min_height = float(max(retry2_cfg.min_height, 2.2))
    retry2_cfg.max_height = float(max(retry2_cfg.max_height + 1.8, retry2_cfg.min_height + 2.6))
    retry2_cfg.height_step = float(min(retry2_cfg.height_step, 0.20))
    retry2_cfg.clash_tau = float(max(0.60, retry2_cfg.clash_tau - 0.05))
    retry2_cfg.site_contact_tolerance = float(retry2_cfg.site_contact_tolerance + 1.20)
    retry2_cfg.n_azimuth = int(max(6, retry2_cfg.n_azimuth))
    retry2_cfg.n_shifts = int(max(2, retry2_cfg.n_shifts))
    retry2_cfg.random_seed = int(retry2_cfg.random_seed) + 23
    return PoseSampler(retry2_cfg).sample(
        slab=slab,
        adsorbate=adsorbate,
        primitives=primitives,
        surface_atom_ids=surface_atom_ids,
    )


def _adsorbate_span(adsorbate: Atoms) -> float:
    if len(adsorbate) <= 1:
        return 0.0
    pos = adsorbate.get_positions()
    return float(max(pos.max(axis=0) - pos.min(axis=0)))


def _make_atomic_number_features(slab: Atoms) -> Any:
    z = slab.get_atomic_numbers().astype(float)
    z = z / (np.max(z) + 1e-12)
    return z.reshape(-1, 1)
