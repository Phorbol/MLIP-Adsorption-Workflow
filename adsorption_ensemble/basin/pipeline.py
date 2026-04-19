from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from adsorption_ensemble.basin.anomaly import classify_anomaly
from adsorption_ensemble.basin.dedup import (
    cluster_by_binding_pattern_and_surface_distance,
    merge_basin_representatives_by_mace_node_l2,
    cluster_by_signature_and_mace_node_l2,
    cluster_by_signature_and_rmsd,
    cluster_by_signature_only,
)
from adsorption_ensemble.selection import apply_stage_selection, stage_selection_summary
from adsorption_ensemble.basin.types import Basin, BasinConfig, BasinResult, RejectedCandidate
from adsorption_ensemble.relax.backends import IdentityRelaxBackend


@dataclass
class BasinBuilder:
    config: BasinConfig
    relax_backend: object | None = None

    def build(
        self,
        frames: list[Atoms],
        slab_ref: Atoms,
        adsorbate_ref: Atoms,
        slab_n: int,
        normal_axis: int = 2,
    ) -> BasinResult:
        cfg = self.config
        backend = self.relax_backend or IdentityRelaxBackend()
        work_dir = cfg.work_dir
        if work_dir is not None:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        relaxed, energies, relax_backend_name = backend.relax(
            frames=frames,
            maxf=float(cfg.relax_maxf),
            steps=int(cfg.relax_steps),
            work_dir=(work_dir / "relax") if work_dir is not None else None,
        )
        rejected: list[RejectedCandidate] = []
        kept_frames: list[Atoms] = []
        kept_energies: list[float] = []
        kept_ids: list[int] = []
        for i, a in enumerate(relaxed):
            reason, metrics = classify_anomaly(
                relaxed=a,
                slab_ref=slab_ref,
                adsorbate_ref=adsorbate_ref,
                slab_n=int(slab_n),
                normal_axis=int(normal_axis),
                binding_tau=float(cfg.binding_tau),
                desorption_min_bonds=int(cfg.desorption_min_bonds),
                desorption_contact_slack=float(cfg.desorption_contact_slack),
                surface_reconstruction_max_disp=float(cfg.surface_reconstruction_max_disp),
                surface_reconstruction_enabled=bool(cfg.surface_reconstruction_enabled),
                dissociation_allow_bond_change=bool(cfg.dissociation_allow_bond_change),
                burial_margin=float(cfg.burial_margin),
            )
            e = float(energies[i]) if i < len(energies) else float("nan")
            metrics = dict(metrics)
            metrics["energy_ev"] = e
            if reason is not None:
                rejected.append(RejectedCandidate(candidate_id=int(i), reason=str(reason), metrics=metrics))
                continue
            kept_frames.append(a)
            kept_energies.append(e)
            kept_ids.append(int(i))
        kept_energies_arr = np.asarray(kept_energies, dtype=float)
        if kept_frames:
            e0 = float(np.nanmin(kept_energies_arr))
        else:
            e0 = float("nan")
        window_keep: list[Atoms] = []
        window_energies: list[float] = []
        window_map_ids: list[int] = []
        for j, a in enumerate(kept_frames):
            e = float(kept_energies_arr[j])
            if np.isfinite(e0) and np.isfinite(e) and e > e0 + float(cfg.energy_window_ev):
                rejected.append(
                    RejectedCandidate(
                        candidate_id=int(kept_ids[j]),
                        reason="energy_window",
                        metrics={"energy_ev": e, "energy_min_ev": e0, "delta_ev": float(e - e0)},
                    )
                )
                continue
            window_keep.append(a)
            window_energies.append(e)
            window_map_ids.append(int(kept_ids[j]))
        post_relax_diag = {
            "enabled": False,
            "strategy": "none",
            "n_input": int(len(window_keep)),
            "n_selected": int(len(window_keep)),
            "selected_ids": list(range(len(window_keep))),
        }
        if window_keep and cfg.post_relax_selection is not None and bool(cfg.post_relax_selection.enabled):
            post_relax_selected_ids, post_relax_diag = apply_stage_selection(
                frames=window_keep,
                config=cfg.post_relax_selection,
                slab_n=int(slab_n),
                energies=np.asarray(window_energies, dtype=float),
                artifacts_dir=((work_dir / "post_relax_selection_rounds") if work_dir is not None else None),
            )
            window_keep = [window_keep[i] for i in post_relax_selected_ids]
            window_energies = [window_energies[i] for i in post_relax_selected_ids]
            window_map_ids = [window_map_ids[i] for i in post_relax_selected_ids]
            if work_dir is not None:
                try:
                    from ase.io import write

                    write((work_dir / "post_relax_selected.extxyz").as_posix(), window_keep)
                except Exception:
                    pass
                try:
                    import json

                    (work_dir / "post_relax_selection.json").write_text(
                        json.dumps(post_relax_diag, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
        dedup_metric = str(cfg.dedup_metric).strip().lower()
        dedup_meta: dict = {}
        if dedup_metric in {"signature", "signature_only"}:
            basins_raw = cluster_by_signature_only(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                signature_mode=str(cfg.signature_mode),
                surface_reference=slab_ref,
            )
        elif dedup_metric in {"binding_surface", "binding_surface_distance", "surface_binding_distance", "binding_surface_descriptor"}:
            basins_raw, dedup_meta = cluster_by_binding_pattern_and_surface_distance(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                surface_distance_threshold=float(cfg.surface_descriptor_threshold),
                surface_nearest_k=int(cfg.surface_descriptor_nearest_k),
                surface_atom_mode=str(cfg.surface_descriptor_atom_mode),
                surface_relative=bool(cfg.surface_descriptor_relative),
                surface_rmsd_gate=cfg.surface_descriptor_rmsd_gate,
                cluster_method=str(cfg.dedup_cluster_method),
                fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
            )
        elif dedup_metric in {"rmsd"}:
            basins_raw = cluster_by_signature_and_rmsd(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                rmsd_threshold=float(cfg.rmsd_threshold),
                cluster_method=str(cfg.dedup_cluster_method),
                fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
                signature_mode=str(cfg.signature_mode),
                use_signature_grouping=True,
                surface_reference=slab_ref,
            )
        elif dedup_metric in {"pure_rmsd", "rmsd_only"}:
            basins_raw = cluster_by_signature_and_rmsd(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                rmsd_threshold=float(cfg.rmsd_threshold),
                cluster_method=str(cfg.dedup_cluster_method),
                fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
                signature_mode=str(cfg.signature_mode),
                use_signature_grouping=False,
                surface_reference=slab_ref,
            )
        else:
            basins_raw = cluster_by_signature_and_rmsd(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                rmsd_threshold=float(cfg.rmsd_threshold),
                cluster_method=str(cfg.dedup_cluster_method),
                fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
                signature_mode=str(cfg.signature_mode),
                use_signature_grouping=True,
                surface_reference=slab_ref,
            )
        if dedup_metric in {"mace", "mace_node_l2", "mace_l2"}:
            basins_raw, dedup_meta = cluster_by_signature_and_mace_node_l2(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                node_l2_threshold=float(cfg.mace_node_l2_threshold),
                mace_model_path=cfg.mace_model_path,
                mace_device=str(cfg.mace_device),
                mace_dtype=str(cfg.mace_dtype),
                mace_enable_cueq=bool(cfg.mace_enable_cueq),
                mace_max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
                mace_layers_to_keep=int(cfg.mace_layers_to_keep),
                mace_head_name=cfg.mace_head_name,
                mace_mlp_energy_key=cfg.mace_mlp_energy_key,
                cluster_method=str(cfg.dedup_cluster_method),
                l2_mode=str(cfg.mace_node_l2_mode),
                fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
                signature_mode=str(cfg.signature_mode),
                use_signature_grouping=True,
                surface_reference=slab_ref,
            )
        if dedup_metric in {"pure_mace", "pure_mace_l2", "mace_only"}:
            basins_raw, dedup_meta = cluster_by_signature_and_mace_node_l2(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                node_l2_threshold=float(cfg.mace_node_l2_threshold),
                mace_model_path=cfg.mace_model_path,
                mace_device=str(cfg.mace_device),
                mace_dtype=str(cfg.mace_dtype),
                mace_enable_cueq=bool(cfg.mace_enable_cueq),
                mace_max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
                mace_layers_to_keep=int(cfg.mace_layers_to_keep),
                mace_head_name=cfg.mace_head_name,
                mace_mlp_energy_key=cfg.mace_mlp_energy_key,
                cluster_method=str(cfg.dedup_cluster_method),
                l2_mode=str(cfg.mace_node_l2_mode),
                fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
                signature_mode=str(cfg.signature_mode),
                use_signature_grouping=False,
                surface_reference=slab_ref,
            )
        final_merge_metric = str(cfg.final_basin_merge_metric).strip().lower()
        final_merge_meta: dict = {
            "enabled": False,
            "metric": str(cfg.final_basin_merge_metric),
            "status": "disabled",
            "n_input_basins": int(len(basins_raw)),
            "n_output_basins": int(len(basins_raw)),
            "energy_gate_ev": (
                None
                if cfg.final_basin_merge_energy_gate_ev is None
                else float(cfg.final_basin_merge_energy_gate_ev)
            ),
        }
        if basins_raw and final_merge_metric not in {"", "off", "none", "disabled"}:
            final_merge_meta["enabled"] = True
            final_merge_meta["n_input_basins"] = int(len(basins_raw))
            ref_canonical_metrics = {
                "reference_canonical_mace",
                "ref_canonical_mace",
                "auto_ref_canonical_mace",
                "auto_reference_canonical_mace",
            }
            threshold = cfg.final_basin_merge_node_l2_threshold
            if threshold is None:
                if final_merge_metric in ref_canonical_metrics:
                    threshold = 0.02
                else:
                    threshold = float(cfg.mace_node_l2_threshold)
            cluster_method = (
                str(cfg.final_basin_merge_cluster_method)
                if cfg.final_basin_merge_cluster_method is not None and str(cfg.final_basin_merge_cluster_method).strip()
                else "hierarchical"
            )
            model_path_use = cfg.mace_model_path
            device_use = str(cfg.mace_device)
            dtype_use = str(cfg.mace_dtype)
            merge_signature_mode = cfg.final_basin_merge_signature_mode
            merge_use_signature_grouping = cfg.final_basin_merge_use_signature_grouping
            if merge_signature_mode is None:
                if final_merge_metric in {"pure_mace", "auto_pure_mace"}:
                    merge_signature_mode = "none"
                elif final_merge_metric in ref_canonical_metrics:
                    merge_signature_mode = "reference_canonical"
                else:
                    merge_signature_mode = "canonical"
            if merge_use_signature_grouping is None:
                merge_use_signature_grouping = final_merge_metric not in {"pure_mace", "auto_pure_mace"}
            if final_merge_metric in ref_canonical_metrics and len(adsorbate_ref) <= 1:
                final_merge_meta.update(
                    {
                        "status": "skipped_single_atom_adsorbate",
                        "n_output_basins": int(len(basins_raw)),
                        "node_l2_threshold": float(threshold),
                        "cluster_method": str(cluster_method),
                        "signature_mode": str(merge_signature_mode),
                        "use_signature_grouping": bool(merge_use_signature_grouping),
                        "energy_gate_ev": (
                            None
                            if cfg.final_basin_merge_energy_gate_ev is None
                            else float(cfg.final_basin_merge_energy_gate_ev)
                        ),
                    }
                )
            elif final_merge_metric in {"auto", "auto_mace", "auto_pure_mace", "auto_ref_canonical_mace", "auto_reference_canonical_mace"}:
                from adsorption_ensemble.relax.backends import normalize_mace_descriptor_config

                model_path_use, device_use, dtype_use = normalize_mace_descriptor_config(
                    model_path=cfg.mace_model_path,
                    device=str(cfg.mace_device),
                    dtype=str(cfg.mace_dtype),
                    strict=False,
                )
                if not model_path_use:
                    final_merge_meta.update(
                        {
                            "status": "skipped_no_model",
                            "n_output_basins": int(len(basins_raw)),
                            "node_l2_threshold": float(threshold),
                            "cluster_method": str(cluster_method),
                            "signature_mode": str(merge_signature_mode),
                            "use_signature_grouping": bool(merge_use_signature_grouping),
                            "energy_gate_ev": (
                                None
                                if cfg.final_basin_merge_energy_gate_ev is None
                                else float(cfg.final_basin_merge_energy_gate_ev)
                            ),
                        }
                    )
            if (
                final_merge_metric
                in {
                    "mace",
                    "mace_node_l2",
                    "mace_l2",
                    "pure_mace",
                    "auto",
                    "auto_mace",
                    "auto_pure_mace",
                    "reference_canonical_mace",
                    "ref_canonical_mace",
                    "auto_ref_canonical_mace",
                    "auto_reference_canonical_mace",
                }
                and (
                    final_merge_metric
                    not in {"auto", "auto_mace", "auto_pure_mace", "auto_ref_canonical_mace", "auto_reference_canonical_mace"}
                    or bool(model_path_use)
                )
                and str(final_merge_meta.get("status", "")) not in {"skipped_single_atom_adsorbate", "skipped_no_model"}
            ):
                try:
                    basins_raw, merge_meta = merge_basin_representatives_by_mace_node_l2(
                        basins=basins_raw,
                        slab_n=int(slab_n),
                        binding_tau=float(cfg.binding_tau),
                        node_l2_threshold=float(threshold),
                        mace_model_path=model_path_use,
                        mace_device=str(device_use),
                        mace_dtype=str(dtype_use),
                        mace_enable_cueq=bool(cfg.mace_enable_cueq),
                        mace_max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
                        mace_layers_to_keep=int(cfg.mace_layers_to_keep),
                        mace_head_name=cfg.mace_head_name,
                        mace_mlp_energy_key=cfg.mace_mlp_energy_key,
                        cluster_method=str(cluster_method),
                        l2_mode=str(cfg.mace_node_l2_mode),
                        fuzzy_sigma_scale=float(cfg.fuzzy_sigma_scale),
                        fuzzy_membership_cutoff=float(cfg.fuzzy_membership_cutoff),
                        signature_mode=str(merge_signature_mode),
                        use_signature_grouping=bool(merge_use_signature_grouping),
                        surface_reference=slab_ref,
                        energy_gate_ev=cfg.final_basin_merge_energy_gate_ev,
                    )
                    backend_metric = merge_meta.get("metric")
                    final_merge_meta.update(dict(merge_meta))
                    final_merge_meta["metric"] = str(cfg.final_basin_merge_metric)
                    if backend_metric is not None:
                        final_merge_meta["backend_metric"] = str(backend_metric)
                    final_merge_meta["status"] = "ok"
                except Exception as exc:
                    if final_merge_metric in {"auto", "auto_mace", "auto_pure_mace"}:
                        final_merge_meta.update(
                            {
                                "status": "error_fallback",
                                "error_type": str(type(exc).__name__),
                                "error_message": str(exc),
                                "n_output_basins": int(len(basins_raw)),
                                "node_l2_threshold": float(threshold),
                                "cluster_method": str(cluster_method),
                                "signature_mode": str(merge_signature_mode),
                                "use_signature_grouping": bool(merge_use_signature_grouping),
                                "energy_gate_ev": (
                                    None
                                    if cfg.final_basin_merge_energy_gate_ev is None
                                    else float(cfg.final_basin_merge_energy_gate_ev)
                                ),
                            }
                        )
                    else:
                        raise
        basins: list[Basin] = []
        for b in basins_raw:
            pairs = list(b["binding_pairs"])
            dent = len({int(i) for i, _ in pairs})
            member_ids = [int(window_map_ids[mid]) for mid in list(b["member_candidate_ids"])]
            basins.append(
                Basin(
                    basin_id=int(b["basin_id"]),
                    atoms=b["atoms"],
                    energy_ev=float(b["energy"]),
                    member_candidate_ids=member_ids,
                    binding_pairs=pairs,
                    denticity=int(dent),
                    signature=str(b["signature"]),
                )
            )
        summary = {
            "n_input": int(len(frames)),
            "n_relaxed": int(len(relaxed)),
            "n_rejected": int(len(rejected)),
            "n_kept": int(len(window_keep)),
            "n_basins": int(len(basins)),
            "energy_min_ev": None if not np.isfinite(e0) else float(e0),
            "dedup_metric": str(cfg.dedup_metric),
            "post_relax_selection": stage_selection_summary(cfg.post_relax_selection),
            "post_relax_selection_diagnostics": dict(post_relax_diag),
            "signature_mode": str(cfg.signature_mode),
            "dedup_cluster_method": str(cfg.dedup_cluster_method),
            "dedup_meta": dict(dedup_meta),
            "final_basin_merge": dict(final_merge_meta),
        }
        return BasinResult(basins=basins, rejected=rejected, relax_backend=str(relax_backend_name), summary=summary)
