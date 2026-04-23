from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.conformer_md import (
    ConformerMDSampler,
    ConformerMDSamplerConfig,
    resolve_selection_profile,
)
from adsorption_ensemble.conformer_md.io_utils import read_molecule_any


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Production-style conformer MD demo using examples/C6H14.gjf.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "examples" / "C6H14.gjf",
        help="ASE-readable molecule input. Defaults to examples/C6H14.gjf.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="adsorption_seed_broad",
        choices=["manual", "isolated_strict", "adsorption_seed_broad"],
        help="Selection profile controlling preselect/final-budget semantics.",
    )
    parser.add_argument(
        "--target-final-k",
        type=int,
        default=None,
        help="Optional override for the final representative budget.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=root / "artifacts" / "c6h14_profile_compare" / "conformer_md",
        help="Output root that will contain the job directory.",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Optional explicit job name.")
    parser.add_argument(
        "--generator-backend",
        type=str,
        default="xtb_md",
        choices=["xtb_md", "rdkit_embed"],
        help="Conformer generator backend.",
    )
    parser.add_argument("--mace-model", type=str, default="/root/.cache/mace/mace-mh-1.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-head-name", type=str, default="omol")
    parser.add_argument("--relax-dtype", type=str, default="float32")
    parser.add_argument("--temperature-k", type=float, default=450.0)
    parser.add_argument("--time-ps", type=float, default=10.0)
    parser.add_argument("--step-fs", type=float, default=1.0)
    parser.add_argument("--dump-fs", type=float, default=50.0)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rdkit-num-confs", type=int, default=96)
    parser.add_argument("--rdkit-prune-rms-thresh", type=float, default=0.15)
    parser.add_argument("--rdkit-embed-method", type=str, default="etkdg_v3")
    parser.add_argument("--rdkit-random-seed", type=int, default=42)
    parser.add_argument("--rdkit-use-random-coords", action="store_true")
    parser.add_argument("--rdkit-num-threads", type=int, default=1)
    parser.add_argument("--rdkit-optimize-forcefield", type=str, default="mmff")
    parser.add_argument("--rdkit-max-opt-iters", type=int, default=200)
    parser.add_argument("--save-all-frames", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> ConformerMDSamplerConfig:
    cfg = ConformerMDSamplerConfig()
    cfg.generator.backend = str(args.generator_backend)
    cfg.md.temperature_k = float(args.temperature_k)
    cfg.md.time_ps = float(args.time_ps)
    cfg.md.step_fs = float(args.step_fs)
    cfg.md.dump_fs = float(args.dump_fs)
    cfg.md.n_runs = int(args.n_runs)
    cfg.md.seed = int(args.seed)
    cfg.md.seed_mode = "increment_per_run"
    cfg.generator.rdkit.num_confs = int(args.rdkit_num_confs)
    cfg.generator.rdkit.prune_rms_thresh = float(args.rdkit_prune_rms_thresh)
    cfg.generator.rdkit.embed_method = str(args.rdkit_embed_method)
    cfg.generator.rdkit.random_seed = int(args.rdkit_random_seed)
    cfg.generator.rdkit.use_random_coords = bool(args.rdkit_use_random_coords)
    cfg.generator.rdkit.num_threads = int(args.rdkit_num_threads)
    cfg.generator.rdkit.optimize_forcefield = str(args.rdkit_optimize_forcefield)
    cfg.generator.rdkit.max_opt_iters = int(args.rdkit_max_opt_iters)
    cfg.selection.mode = "fps_pca_kmeans"
    cfg.selection.structure_metric_threshold = 0.05
    cfg.selection.pca_variance_threshold = 0.95
    cfg.selection.fps_pool_factor = 3
    cfg.descriptor.backend = "mace"
    cfg.relax.backend = "mace_relax"
    cfg.descriptor.mace.model_path = str(args.mace_model)
    cfg.relax.mace.model_path = str(args.mace_model)
    cfg.descriptor.mace.head_name = str(args.mace_head_name)
    cfg.relax.mace.head_name = str(args.mace_head_name)
    cfg.descriptor.mace.device = str(args.mace_device)
    cfg.relax.mace.device = str(args.mace_device)
    cfg.descriptor.mace.dtype = "float64"
    cfg.relax.mace.dtype = str(args.relax_dtype)
    cfg.descriptor.mace.enable_cueq = False
    cfg.relax.mace.enable_cueq = False
    cfg.descriptor.mace.max_edges_per_batch = 15000
    cfg.relax.mace.max_edges_per_batch = 15000
    cfg.relax.loose.maxf = 0.5
    cfg.relax.loose.steps = 50
    cfg.relax.refine.maxf = 0.05
    cfg.relax.refine.steps = 100
    cfg.output.work_dir = Path(args.work_root)
    cfg.output.save_all_frames = bool(args.save_all_frames)
    return resolve_selection_profile(
        cfg,
        profile=str(args.profile),
        target_final_k=args.target_final_k,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    atoms = read_molecule_any(input_path)
    cfg = build_config(args)
    job_name = str(args.job_name or f"{input_path.stem}__{cfg.selection.selection_profile}")
    case_dir = cfg.output.work_dir / job_name
    case_dir.mkdir(parents=True, exist_ok=True)
    write((case_dir / f"{input_path.stem}.xyz").as_posix(), atoms)
    sampler = ConformerMDSampler(config=cfg)
    result = sampler.run(atoms, job_name=job_name)
    print("Conformer search finished.")
    print(f"input={input_path}")
    print(f"generator_backend={cfg.generator.backend}")
    print(f"profile={cfg.selection.selection_profile}")
    print(f"target_final_k={cfg.selection.target_final_k}")
    print(f"n_selected={len(result.conformers)}")
    print(f"work_dir={case_dir}")
    print(f"summary={case_dir / 'summary.json'}")
    print(f"metadata={case_dir / 'metadata.json'}")
    print(f"ensemble={case_dir / 'ensemble.extxyz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
