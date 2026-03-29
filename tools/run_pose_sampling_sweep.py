from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.pose.sweep import PoseSweepConfig, run_pose_sampling_sweep, summarize_rows, summary_to_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/pose_sampling")
    parser.add_argument("--slab-name", type=str, default="")
    parser.add_argument("--max-molecules", type=int, default=30)
    parser.add_argument("--max-slabs", type=int, default=20)
    parser.add_argument("--max-atoms-per-molecule", type=int, default=12)
    parser.add_argument("--max-combinations", type=int, default=400)
    parser.add_argument("--grid-step", type=float, default=0.6)
    parser.add_argument("--spacing", type=float, default=0.8)
    parser.add_argument("--l2-threshold", type=float, default=0.22)
    parser.add_argument("--max-basis-sites", type=int, default=-1)
    parser.add_argument("--n-rotations", type=int, default=8)
    parser.add_argument("--n-azimuth", type=int, default=12)
    parser.add_argument("--n-shifts", type=int, default=3)
    parser.add_argument("--shift-radius", type=float, default=0.25)
    parser.add_argument("--n-height-shifts", type=int, default=1)
    parser.add_argument("--height-shift-step", type=float, default=0.0)
    parser.add_argument("--min-height", type=float, default=1.0)
    parser.add_argument("--max-height", type=float, default=3.4)
    parser.add_argument("--height-step", type=float, default=0.08)
    parser.add_argument("--height-taus", type=str, default="0.90,0.95,1.00")
    parser.add_argument("--site-contact-tolerance", type=float, default=0.20)
    parser.add_argument("--clash-tau", type=float, default=0.85)
    parser.add_argument("--max-poses-per-site", type=int, default=24)
    parser.add_argument("--max-poses-output", type=int, default=256)
    parser.add_argument("--postprocess-enabled", action="store_true")
    parser.add_argument("--postprocess-disabled", action="store_true")
    parser.add_argument("--postprocess-preselect-k", type=int, default=64)
    parser.add_argument("--postprocess-loose-maxf", type=float, default=0.25)
    parser.add_argument("--postprocess-loose-steps", type=int, default=20)
    parser.add_argument("--postprocess-refine-maxf", type=float, default=0.08)
    parser.add_argument("--postprocess-refine-steps", type=int, default=60)
    parser.add_argument("--postprocess-energy-window-ev", type=float, default=3.0)
    parser.add_argument("--postprocess-rmsd-threshold", type=float, default=0.08)
    parser.add_argument("--postprocess-final-energy-window-ev", type=float, default=3.0)
    parser.add_argument("--postprocess-final-rmsd-threshold", type=float, default=0.05)
    parser.add_argument("--mace-model-path", type=str, default="")
    parser.add_argument("--mace-desc-device", type=str, default="cuda")
    parser.add_argument("--mace-desc-dtype", type=str, default="float64")
    parser.add_argument("--mace-relax-device", type=str, default="cuda")
    parser.add_argument("--mace-relax-dtype", type=str, default="float32")
    parser.add_argument("--mace-max-edges-per-batch", type=int, default=15000)
    parser.add_argument("--mace-head-name", type=str, default="")
    parser.add_argument("--mace-strict", action="store_true")
    parser.add_argument("--postprocess-batch-relax-enabled", action="store_true")
    parser.add_argument("--postprocess-batch-relax-disabled", action="store_true")
    parser.add_argument("--neighborlist-disabled", action="store_true")
    parser.add_argument("--neighborlist-min-surface-atoms", type=int, default=64)
    parser.add_argument("--neighborlist-cutoff-padding", type=float, default=0.30)
    parser.add_argument("--ensemble-enabled", action="store_true")
    parser.add_argument("--ensemble-disabled", action="store_true")
    parser.add_argument("--ensemble-relax-backend", type=str, default="mace_batch")
    parser.add_argument("--ensemble-relax-maxf", type=float, default=0.10)
    parser.add_argument("--ensemble-relax-steps", type=int, default=80)
    parser.add_argument("--ensemble-energy-window-ev", type=float, default=3.0)
    parser.add_argument("--ensemble-dedup-metric", type=str, default="rmsd")
    parser.add_argument("--ensemble-rmsd-threshold", type=float, default=0.10)
    parser.add_argument("--ensemble-mace-node-l2-threshold", type=float, default=2.0)
    parser.add_argument("--ensemble-binding-tau", type=float, default=1.15)
    parser.add_argument("--ensemble-desorption-min-bonds", type=int, default=1)
    parser.add_argument("--ensemble-surface-reconstruction-max-disp", type=float, default=0.50)
    parser.add_argument("--ensemble-dissociation-allow-bond-change", action="store_true")
    parser.add_argument("--ensemble-burial-margin", type=float, default=0.30)
    parser.add_argument("--node-bond-tau", type=float, default=1.20)
    parser.add_argument("--node-hash-len", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--profiling-enabled", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--cprofile-out", type=str, default="")
    args = parser.parse_args()
    taus = tuple(float(x.strip()) for x in args.height_taus.split(",") if x.strip())
    post_enabled = True
    if args.postprocess_enabled:
        post_enabled = True
    if args.postprocess_disabled:
        post_enabled = False
    batch_relax_enabled = True
    if bool(args.postprocess_batch_relax_enabled):
        batch_relax_enabled = True
    if bool(args.postprocess_batch_relax_disabled):
        batch_relax_enabled = False
    ensemble_enabled = False
    if bool(args.ensemble_enabled):
        ensemble_enabled = True
    if bool(args.ensemble_disabled):
        ensemble_enabled = False
    cfg = PoseSweepConfig(
        grid_step=args.grid_step,
        spacing=args.spacing,
        l2_distance_threshold=args.l2_threshold,
        max_basis_sites=(None if args.max_basis_sites < 0 else int(args.max_basis_sites)),
        n_rotations=args.n_rotations,
        n_azimuth=args.n_azimuth,
        n_shifts=args.n_shifts,
        shift_radius=args.shift_radius,
        n_height_shifts=args.n_height_shifts,
        height_shift_step=args.height_shift_step,
        min_height=args.min_height,
        max_height=args.max_height,
        height_step=args.height_step,
        height_taus=taus,
        site_contact_tolerance=args.site_contact_tolerance,
        clash_tau=args.clash_tau,
        max_poses_per_site=args.max_poses_per_site,
        postprocess_enabled=post_enabled,
        postprocess_preselect_k=args.postprocess_preselect_k,
        postprocess_loose_maxf=args.postprocess_loose_maxf,
        postprocess_loose_steps=args.postprocess_loose_steps,
        postprocess_refine_maxf=args.postprocess_refine_maxf,
        postprocess_refine_steps=args.postprocess_refine_steps,
        postprocess_energy_window_ev=args.postprocess_energy_window_ev,
        postprocess_rmsd_threshold=args.postprocess_rmsd_threshold,
        postprocess_final_energy_window_ev=args.postprocess_final_energy_window_ev,
        postprocess_final_rmsd_threshold=args.postprocess_final_rmsd_threshold,
        mace_model_path=args.mace_model_path,
        mace_desc_device=args.mace_desc_device,
        mace_desc_dtype=args.mace_desc_dtype,
        mace_relax_device=args.mace_relax_device,
        mace_relax_dtype=args.mace_relax_dtype,
        mace_max_edges_per_batch=int(args.mace_max_edges_per_batch),
        mace_head_name=str(args.mace_head_name),
        mace_strict=bool(args.mace_strict),
        postprocess_batch_relax_enabled=bool(batch_relax_enabled),
        profiling_enabled=bool(args.profiling_enabled),
        neighborlist_enabled=(not bool(args.neighborlist_disabled)),
        neighborlist_min_surface_atoms=int(args.neighborlist_min_surface_atoms),
        neighborlist_cutoff_padding=float(args.neighborlist_cutoff_padding),
        ensemble_enabled=bool(ensemble_enabled),
        ensemble_relax_backend=str(args.ensemble_relax_backend),
        ensemble_relax_maxf=float(args.ensemble_relax_maxf),
        ensemble_relax_steps=int(args.ensemble_relax_steps),
        ensemble_energy_window_ev=float(args.ensemble_energy_window_ev),
        ensemble_dedup_metric=str(args.ensemble_dedup_metric),
        ensemble_rmsd_threshold=float(args.ensemble_rmsd_threshold),
        ensemble_mace_node_l2_threshold=float(args.ensemble_mace_node_l2_threshold),
        ensemble_binding_tau=float(args.ensemble_binding_tau),
        ensemble_desorption_min_bonds=int(args.ensemble_desorption_min_bonds),
        ensemble_surface_reconstruction_max_disp=float(args.ensemble_surface_reconstruction_max_disp),
        ensemble_dissociation_allow_bond_change=bool(args.ensemble_dissociation_allow_bond_change),
        ensemble_burial_margin=float(args.ensemble_burial_margin),
        node_bond_tau=float(args.node_bond_tau),
        node_hash_len=int(args.node_hash_len),
        random_seed=args.random_seed,
        max_poses_output=args.max_poses_output,
    )
    slab_filter = [str(args.slab_name)] if str(args.slab_name).strip() else None
    if bool(args.cprofile):
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
        out = run_pose_sampling_sweep(
            out_root=Path(args.out_root),
            cfg=cfg,
            max_molecules=args.max_molecules,
            max_slabs=args.max_slabs,
            max_atoms_per_molecule=args.max_atoms_per_molecule,
            max_combinations=args.max_combinations,
            slab_filter=slab_filter,
        )
        profiler.disable()
        out_path = Path(args.cprofile_out) if str(args.cprofile_out).strip() else Path(out["run_dir"]) / "cprofile.stats"
        profiler.dump_stats(out_path.as_posix())
        print(json.dumps({"cprofile_stats": out_path.as_posix()}, ensure_ascii=False, indent=2))
    else:
        out = run_pose_sampling_sweep(
            out_root=Path(args.out_root),
            cfg=cfg,
            max_molecules=args.max_molecules,
            max_slabs=args.max_slabs,
            max_atoms_per_molecule=args.max_atoms_per_molecule,
            max_combinations=args.max_combinations,
            slab_filter=slab_filter,
        )
    summary = summarize_rows(out["rows"])
    print(summary_to_text(summary))
    print(json.dumps({"run_dir": out["run_dir"], "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
