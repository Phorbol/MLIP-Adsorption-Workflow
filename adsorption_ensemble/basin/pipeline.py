from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from adsorption_ensemble.basin.anomaly import classify_anomaly
from adsorption_ensemble.basin.dedup import (
    cluster_by_signature_and_mace_node_l2,
    cluster_by_signature_and_rmsd,
    cluster_by_signature_only,
)
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
                surface_reconstruction_max_disp=float(cfg.surface_reconstruction_max_disp),
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
        dedup_metric = str(cfg.dedup_metric).strip().lower()
        if dedup_metric in {"signature", "signature_only"}:
            basins_raw = cluster_by_signature_only(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
            )
        else:
            basins_raw = cluster_by_signature_and_rmsd(
                frames=window_keep,
                energies=np.asarray(window_energies, dtype=float),
                slab_n=int(slab_n),
                binding_tau=float(cfg.binding_tau),
                rmsd_threshold=float(cfg.rmsd_threshold),
            )
        dedup_meta: dict = {}
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
                mace_max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
                mace_layers_to_keep=int(cfg.mace_layers_to_keep),
                mace_head_name=cfg.mace_head_name,
                mace_mlp_energy_key=cfg.mace_mlp_energy_key,
            )
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
            "dedup_meta": dict(dedup_meta),
        }
        return BasinResult(basins=basins, rejected=rejected, relax_backend=str(relax_backend_name), summary=summary)
