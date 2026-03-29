from __future__ import annotations

import argparse

from .config import ConformerMDSamplerConfig
from .io_utils import read_molecule_any
from .pipeline import ConformerMDSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conformer-md")
    parser.add_argument("input_xyz", type=str)
    parser.add_argument("--job-name", type=str, default="conformer_job")
    parser.add_argument("--temperature-k", type=float, default=400.0)
    parser.add_argument("--time-ps", type=float, default=100.0)
    parser.add_argument("--step-fs", type=float, default=1.0)
    parser.add_argument("--dump-fs", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--preselect-k", type=int, default=64)
    parser.add_argument("--mode", type=str, default="fps_pca_kmeans")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.95)
    parser.add_argument("--fps-pool-factor", type=int, default=3)
    parser.add_argument("--energy-window-ev", type=float, default=0.2)
    parser.add_argument("--rmsd-threshold", type=float, default=0.05)
    parser.add_argument("--descriptor-backend", type=str, default="geometry")
    parser.add_argument("--relax-backend", type=str, default="mace_relax")
    parser.add_argument("--mace-model", type=str, default=None)
    parser.add_argument("--mace-device", type=str, default="cpu")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--mace-max-edges", type=int, default=15000)
    parser.add_argument("--mace-workers", type=int, default=1)
    parser.add_argument("--mace-layers-to-keep", type=int, default=-1)
    parser.add_argument("--mace-energy-key", type=str, default=None)
    parser.add_argument("--mace-head-name", type=str, default="Default")
    parser.add_argument("--loose-maxf", type=float, default=0.5)
    parser.add_argument("--loose-steps", type=int, default=50)
    parser.add_argument("--refine-maxf", type=float, default=0.05)
    parser.add_argument("--refine-steps", type=int, default=100)
    parser.add_argument("--save-all-frames", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = ConformerMDSamplerConfig()
    cfg.md.temperature_k = args.temperature_k
    cfg.md.time_ps = args.time_ps
    cfg.md.step_fs = args.step_fs
    cfg.md.dump_fs = args.dump_fs
    cfg.md.seed = args.seed
    cfg.md.n_runs = args.n_runs
    cfg.selection.preselect_k = args.preselect_k
    cfg.selection.mode = args.mode
    cfg.selection.pca_variance_threshold = args.pca_variance_threshold
    cfg.selection.fps_pool_factor = args.fps_pool_factor
    cfg.selection.energy_window_ev = args.energy_window_ev
    cfg.selection.rmsd_threshold = args.rmsd_threshold
    cfg.descriptor.backend = args.descriptor_backend
    cfg.relax.backend = args.relax_backend
    cfg.descriptor.mace.model_path = args.mace_model
    cfg.relax.mace.model_path = args.mace_model
    cfg.descriptor.mace.device = args.mace_device
    cfg.relax.mace.device = args.mace_device
    cfg.descriptor.mace.dtype = args.mace_dtype
    cfg.relax.mace.dtype = args.mace_dtype
    cfg.descriptor.mace.max_edges_per_batch = args.mace_max_edges
    cfg.relax.mace.max_edges_per_batch = args.mace_max_edges
    cfg.descriptor.mace.num_workers = args.mace_workers
    cfg.relax.mace.num_workers = args.mace_workers
    cfg.descriptor.mace.layers_to_keep = args.mace_layers_to_keep
    cfg.relax.mace.layers_to_keep = args.mace_layers_to_keep
    cfg.descriptor.mace.mlp_energy_key = args.mace_energy_key
    cfg.relax.mace.mlp_energy_key = args.mace_energy_key
    cfg.descriptor.mace.head_name = args.mace_head_name
    cfg.relax.mace.head_name = args.mace_head_name
    cfg.relax.loose.maxf = args.loose_maxf
    cfg.relax.loose.steps = args.loose_steps
    cfg.relax.refine.maxf = args.refine_maxf
    cfg.relax.refine.steps = args.refine_steps
    cfg.output.save_all_frames = bool(args.save_all_frames)
    mol = read_molecule_any(args.input_xyz)
    sampler = ConformerMDSampler(config=cfg)
    result = sampler.run(molecule=mol, job_name=args.job_name)
    run_dir = cfg.output.work_dir / args.job_name
    print(f"run_dir={run_dir}")
    print(f"n_selected={len(result.conformers)}")
    print(f"summary={run_dir / 'summary.txt'}")
    print(f"metadata={run_dir / 'metadata.json'}")
    print(f"ensemble={run_dir / 'ensemble.extxyz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
