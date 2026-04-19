from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.build import bcc110, bulk, fcc100, fcc111, fcc211, surface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin.dedup import cluster_by_signature_and_mace_node_l2
from adsorption_ensemble.basin.pipeline import BasinBuilder
from adsorption_ensemble.basin.types import BasinConfig
from adsorption_ensemble.conformer_md.descriptors import GeometryPairDistanceDescriptor
from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.pose.postprocess import run_iterative_pose_fps_preselection
from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig
from adsorption_ensemble.workflows import make_default_surface_preprocessor
from tests.chemistry_cases import get_test_adsorbate_cases


def build_slabs() -> dict[str, Atoms]:
    return {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=10.0),
        "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=10.0),
        "bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=10.0),
        "cu321": surface(bulk("Cu", "fcc", a=3.6, cubic=True), (3, 2, 1), layers=4, vacuum=10.0).repeat((2, 2, 1)),
    }


def infer_head_name(model_path: str | None) -> str | None:
    if not model_path:
        return None
    name = Path(model_path).name.lower()
    if "omat" in name:
        return "omat_pbe"
    if "omol" in name:
        return "omol"
    return None


def _adsorbate_span(adsorbate: Atoms) -> float:
    if len(adsorbate) <= 1:
        return 0.0
    pos = adsorbate.get_positions()
    return float(max(pos.max(axis=0) - pos.min(axis=0)))


def _sample_with_fallback(
    sampler: PoseSampler,
    slab: Atoms,
    adsorbate: Atoms,
    primitives: list,
    surface_atom_ids: list[int],
) -> list:
    poses = sampler.sample(slab=slab, adsorbate=adsorbate, primitives=primitives, surface_atom_ids=surface_atom_ids)
    if poses:
        return poses
    retry_cfg = PoseSamplerConfig(**vars(sampler.config))
    span = _adsorbate_span(adsorbate)
    retry_cfg.min_height = float(max(retry_cfg.min_height, 1.8))
    retry_cfg.max_height = float(max(retry_cfg.max_height + 0.8, retry_cfg.min_height + 1.0 + 0.15 * span))
    retry_cfg.height_step = float(min(retry_cfg.height_step, 0.15))
    retry_cfg.clash_tau = float(max(0.65, retry_cfg.clash_tau - 0.1))
    retry_cfg.site_contact_tolerance = float(retry_cfg.site_contact_tolerance + 0.15)
    retry_cfg.random_seed = int(retry_cfg.random_seed) + 17
    retry = PoseSampler(retry_cfg)
    return retry.sample(slab=slab, adsorbate=adsorbate, primitives=primitives, surface_atom_ids=surface_atom_ids)


def build_pose_pool(
    *,
    slab: Atoms,
    adsorbate: Atoms,
    max_selected_primitives: int,
    pose_cfg: PoseSamplerConfig,
) -> tuple[list[Atoms], int, int]:
    pre = make_default_surface_preprocessor()
    ctx = pre.build_context(slab)
    raw_primitives = PrimitiveBuilder().build(slab, ctx)
    z = slab.get_atomic_numbers().astype(float)
    atom_features = (z / (np.max(z) + 1e-12)).reshape(-1, 1)
    embed = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.22)).fit_transform(
        slab=slab, primitives=raw_primitives, atom_features=atom_features
    )
    # Use inequivalent primitives (basis representatives) to avoid oversampling equivalent top sites.
    primitives = list(embed.basis_primitives)[: max(1, int(max_selected_primitives))]
    sampler = PoseSampler(pose_cfg)
    poses = _sample_with_fallback(
        sampler=sampler,
        slab=slab,
        adsorbate=adsorbate,
        primitives=primitives,
        surface_atom_ids=ctx.detection.surface_atom_ids,
    )
    frames: list[Atoms] = []
    for p in poses:
        frame = slab + p.atoms
        frames.append(frame)
    return frames, len(slab), int(ctx.classification.normal_axis)


def fps_subsample(
    *,
    case_out: Path,
    frames: list[Atoms],
    random_seed: int,
    k: int,
) -> list[Atoms]:
    if not frames:
        return []
    desc = GeometryPairDistanceDescriptor(use_float64=False)
    feats = np.asarray(desc.transform(frames), dtype=float)
    out = run_iterative_pose_fps_preselection(
        case_out=case_out,
        features=feats,
        pooled=frames,
        random_seed=int(random_seed),
        k=int(k),
        grid_convergence=True,
        grid_convergence_pca_var=0.95,
        grid_convergence_grid_bins=12,
        grid_convergence_min_rounds=3,
        grid_convergence_patience=2,
        grid_convergence_min_coverage_gain=1e-3,
        grid_convergence_min_novelty=5e-2,
    )
    ids = out["selected_ids"]
    return [frames[i] for i in ids]


def build_basins(
    *,
    frames: list[Atoms],
    slab_ref: Atoms,
    adsorbate_ref: Atoms,
    slab_n: int,
    normal_axis: int,
    cfg: BasinConfig,
    relax_backend,
):
    return BasinBuilder(config=cfg, relax_backend=relax_backend).build(
        frames=frames,
        slab_ref=slab_ref,
        adsorbate_ref=adsorbate_ref,
        slab_n=slab_n,
        normal_axis=normal_axis,
    )


def strategy_score(*, n_pose: int, n_basins: int, runtime_sec: float) -> float:
    if n_pose <= 0:
        return 0.0
    nonzero = 1.0 if n_basins > 0 else 0.0
    ratio = float(n_basins) / float(max(1, n_pose))
    ratio_term = max(0.0, 1.0 - abs(ratio - 0.65) / 0.65)
    runtime_term = max(0.0, 1.0 - runtime_sec / 180.0)
    return 100.0 * (0.60 * nonzero + 0.30 * ratio_term + 0.10 * runtime_term)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/autoresearch/strategy_selection")
    parser.add_argument("--mace-model-path", type=str, required=True)
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--energy-window-list", type=str, default="2.0,3.0,5.0")
    parser.add_argument("--distance-threshold-list", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--max-selected-primitives", type=int, default=12)
    parser.add_argument("--fps-k", type=int, default=48)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()

    if str(args.mace_device).lower().startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slabs = build_slabs()
    ads = get_test_adsorbate_cases()
    case_list = [
        ("fcc111", "CO"),
        ("fcc100", "H2O"),
        ("fcc211", "CH3OH"),
        ("bcc110", "C2H4"),
        ("cu321", "C6H6"),
        ("fcc111", "glucose_chain_like"),
        ("fcc100", "glucose_ring_like"),
        ("fcc211", "glycine_like"),
    ][: max(1, int(args.max_cases))]
    ewindows = [float(x) for x in str(args.energy_window_list).split(",") if str(x).strip()]
    thr_list = [float(x) for x in str(args.distance_threshold_list).split(",") if str(x).strip()]
    pose_cfg = PoseSamplerConfig(
        n_rotations=2,
        n_azimuth=4,
        n_shifts=1,
        shift_radius=0.2,
        min_height=1.5,
        max_height=3.0,
        height_step=0.2,
        max_poses_per_site=2,
        random_seed=0,
    )
    head_name = infer_head_name(str(args.mace_model_path))
    loose_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=str(args.mace_model_path),
            device=str(args.mace_device),
            dtype=str(args.mace_dtype),
            max_edges_per_batch=20000,
            head_name=head_name,
            strict=True,
        )
    )
    strategy_rows = []
    for slab_name, ads_name in case_list:
        case_dir = out_root / slab_name / ads_name
        case_dir.mkdir(parents=True, exist_ok=True)
        slab = slabs[slab_name]
        adsorbate = ads[ads_name]
        frames, slab_n, normal_axis = build_pose_pool(
            slab=slab,
            adsorbate=adsorbate,
            max_selected_primitives=int(args.max_selected_primitives),
            pose_cfg=pose_cfg,
        )
        fps_frames = fps_subsample(case_out=case_dir, frames=frames, random_seed=0, k=min(len(frames), int(args.fps_k)))
        for ewin in ewindows:
            for thr in thr_list:
                for strategy in (
                    "fps_loose_rmsd_window",
                    "fps_loose_mace_window",
                    "fps_loose_rmsd_hier_window",
                    "fps_loose_mace_hier_window",
                    "fps_loose_rmsd_fuzzy_window",
                    "fps_loose_mace_fuzzy_window",
                ):
                    is_mace = "_mace_" in strategy or strategy.endswith("_mace_window")
                    cluster_method = "greedy"
                    if "_hier_" in strategy:
                        cluster_method = "hierarchical"
                    elif "_fuzzy_" in strategy:
                        cluster_method = "fuzzy"
                    cfg = BasinConfig(
                        relax_maxf=0.15,
                        relax_steps=20,
                        energy_window_ev=float(ewin),
                        dedup_metric=("mace_node_l2" if is_mace else "rmsd"),
                        dedup_cluster_method=str(cluster_method),
                        rmsd_threshold=float(thr),
                        mace_node_l2_threshold=float(thr),
                        mace_node_l2_mode="mean_atom",
                        mace_model_path=str(args.mace_model_path),
                        mace_device=str(args.mace_device),
                        mace_dtype=str(args.mace_dtype),
                        mace_head_name=head_name,
                        desorption_min_bonds=0,
                        surface_reconstruction_max_disp=0.50,
                        dissociation_allow_bond_change=False,
                        burial_margin=0.30,
                        work_dir=case_dir / f"tmp_{strategy}_ew{ewin}_thr{thr}",
                    )
                    t0 = time.perf_counter()
                    res = build_basins(
                        frames=fps_frames,
                        slab_ref=slab,
                        adsorbate_ref=adsorbate,
                        slab_n=slab_n,
                        normal_axis=normal_axis,
                        cfg=cfg,
                        relax_backend=loose_backend,
                    )
                    elapsed = time.perf_counter() - t0
                    score = strategy_score(n_pose=len(fps_frames), n_basins=len(res.basins), runtime_sec=elapsed)
                    strategy_rows.append(
                        {
                            "slab": slab_name,
                            "adsorbate": ads_name,
                            "strategy": strategy,
                            "energy_window_ev": float(ewin),
                            "distance_threshold": float(thr),
                            "rmsd_threshold": float(thr),
                            "mace_node_l2_threshold": float(thr),
                            "n_pose_frames": int(len(fps_frames)),
                            "n_basins": int(len(res.basins)),
                            "n_rejected": int(len(res.rejected)),
                            "runtime_sec": float(elapsed),
                            "score": float(score),
                        }
                    )
                # two-stage: loose+rmsd then fine mace split
                cfg_stage1 = BasinConfig(
                    relax_maxf=0.15,
                    relax_steps=20,
                    energy_window_ev=float(ewin),
                    dedup_metric="rmsd",
                    dedup_cluster_method="greedy",
                    rmsd_threshold=float(thr),
                    mace_node_l2_threshold=float(thr),
                    mace_node_l2_mode="mean_atom",
                    mace_model_path=str(args.mace_model_path),
                    mace_device=str(args.mace_device),
                    mace_dtype=str(args.mace_dtype),
                    mace_head_name=head_name,
                    desorption_min_bonds=0,
                    surface_reconstruction_max_disp=0.50,
                    dissociation_allow_bond_change=False,
                    burial_margin=0.30,
                    work_dir=case_dir / f"tmp_fps_loose_rmsd_then_mace_ew{ewin}_thr{thr}",
                )
                t0 = time.perf_counter()
                stage1 = build_basins(
                    frames=fps_frames,
                    slab_ref=slab,
                    adsorbate_ref=adsorbate,
                    slab_n=slab_n,
                    normal_axis=normal_axis,
                    cfg=cfg_stage1,
                    relax_backend=loose_backend,
                )
                stage1_frames = [b.atoms for b in stage1.basins]
                stage1_energies = np.asarray([b.energy_ev for b in stage1.basins], dtype=float)
                if stage1_frames:
                    bas2, _ = cluster_by_signature_and_mace_node_l2(
                        frames=stage1_frames,
                        energies=stage1_energies,
                        slab_n=slab_n,
                        binding_tau=float(cfg_stage1.binding_tau),
                        node_l2_threshold=float(thr),
                        mace_model_path=str(args.mace_model_path),
                        mace_device=str(args.mace_device),
                        mace_dtype=str(args.mace_dtype),
                        mace_max_edges_per_batch=int(cfg_stage1.mace_max_edges_per_batch),
                        mace_layers_to_keep=int(cfg_stage1.mace_layers_to_keep),
                        mace_head_name=cfg_stage1.mace_head_name,
                        mace_mlp_energy_key=cfg_stage1.mace_mlp_energy_key,
                        cluster_method="greedy",
                        l2_mode="mean_atom",
                    )
                    n_basins = len(bas2)
                else:
                    n_basins = 0
                elapsed = time.perf_counter() - t0
                strategy_rows.append(
                    {
                        "slab": slab_name,
                        "adsorbate": ads_name,
                        "strategy": "fps_loose_rmsd_then_mace_window",
                        "energy_window_ev": float(ewin),
                        "distance_threshold": float(thr),
                        "rmsd_threshold": float(thr),
                        "mace_node_l2_threshold": float(thr),
                        "n_pose_frames": int(len(fps_frames)),
                        "n_basins": int(n_basins),
                        "n_rejected": int(len(stage1.rejected)),
                        "runtime_sec": float(elapsed),
                        "score": float(strategy_score(n_pose=len(fps_frames), n_basins=int(n_basins), runtime_sec=elapsed)),
                    }
                )

    by_key = {}
    for r in strategy_rows:
        key = (r["strategy"], r["energy_window_ev"], r["distance_threshold"])
        by_key.setdefault(key, []).append(r)
    ranking = []
    for k, items in by_key.items():
        ranking.append(
            {
                "strategy": k[0],
                "energy_window_ev": k[1],
                "distance_threshold": k[2],
                "mace_node_l2_threshold": k[2],
                "rmsd_threshold": k[2],
                "mean_score": float(np.mean([float(x["score"]) for x in items])),
                "nonzero_case_rate": float(np.mean([1.0 if int(x["n_basins"]) > 0 else 0.0 for x in items])),
                "mean_runtime_sec": float(np.mean([float(x["runtime_sec"]) for x in items])),
                "mean_basins": float(np.mean([int(x["n_basins"]) for x in items])),
                "n_cases": int(len(items)),
            }
        )
    ranking = sorted(ranking, key=lambda x: (-x["mean_score"], -x["nonzero_case_rate"], x["mean_runtime_sec"]))
    payload = {
        "n_case_runs": len(strategy_rows),
        "case_list": case_list,
        "rows": strategy_rows,
        "ranking": ranking,
        "best": (ranking[0] if ranking else None),
    }
    (out_root / "strategy_selection_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": payload["best"], "n_case_runs": payload["n_case_runs"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
