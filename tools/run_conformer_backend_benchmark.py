from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, read_molecule_any, resolve_selection_profile


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark xtb_md vs rdkit_embed under a shared conformer pipeline.")
    p.add_argument("input", type=str)
    p.add_argument("--out-root", type=str, default="artifacts/conformer_backend_benchmark")
    p.add_argument("--backends", type=str, default="xtb_md,rdkit_embed")
    p.add_argument("--selection-profile", type=str, default="manual")
    p.add_argument("--target-final-k", type=int, default=8)
    p.add_argument("--descriptor-backend", type=str, default="geometry")
    p.add_argument("--metric-backend", type=str, default="auto")
    p.add_argument("--relax-backend", type=str, default="identity")
    p.add_argument("--mace-model", type=str, default=None)
    p.add_argument("--mace-device", type=str, default="cpu")
    p.add_argument("--mace-dtype", type=str, default="float32")
    p.add_argument("--mace-head-name", type=str, default="Default")
    p.add_argument("--energy-semantics", type=str, choices=["total", "per_atom"], default="total")
    p.add_argument("--structure-metric-threshold", type=float, default=0.05)
    p.add_argument("--pair-energy-gap-ev", type=float, default=0.0)
    p.add_argument("--xtb-time-ps", type=float, default=10.0)
    p.add_argument("--xtb-n-runs", type=int, default=2)
    p.add_argument("--rdkit-num-confs", type=int, default=128)
    p.add_argument("--rdkit-prune-rms-thresh", type=float, default=0.25)
    p.add_argument("--rdkit-optimize-forcefield", type=str, default="mmff")
    p.add_argument("--save-all-frames", action="store_true")
    return p


def make_config(args, *, backend: str, out_root: Path) -> ConformerMDSamplerConfig:
    cfg = ConformerMDSamplerConfig()
    cfg.generator.backend = str(backend)
    cfg.descriptor.backend = str(args.descriptor_backend)
    cfg.selection.metric_backend = str(args.metric_backend)
    cfg.relax.backend = str(args.relax_backend)
    cfg.descriptor.mace.model_path = args.mace_model
    cfg.relax.mace.model_path = args.mace_model
    cfg.descriptor.mace.device = str(args.mace_device)
    cfg.relax.mace.device = str(args.mace_device)
    cfg.descriptor.mace.dtype = str(args.mace_dtype)
    cfg.relax.mace.dtype = str(args.mace_dtype)
    cfg.descriptor.mace.head_name = str(args.mace_head_name)
    cfg.relax.mace.head_name = str(args.mace_head_name)
    cfg.selection.use_total_energy = bool(str(args.energy_semantics).strip().lower() == "total")
    cfg.selection.structure_metric_threshold = float(args.structure_metric_threshold)
    cfg.selection.pair_energy_gap_ev = float(args.pair_energy_gap_ev)
    cfg.output.save_all_frames = bool(args.save_all_frames)
    cfg.output.work_dir = out_root
    cfg.md.time_ps = float(args.xtb_time_ps)
    cfg.md.n_runs = int(args.xtb_n_runs)
    cfg.generator.rdkit.num_confs = int(args.rdkit_num_confs)
    cfg.generator.rdkit.prune_rms_thresh = float(args.rdkit_prune_rms_thresh)
    cfg.generator.rdkit.optimize_forcefield = str(args.rdkit_optimize_forcefield)
    if str(args.selection_profile).strip().lower() not in {"", "manual"}:
        cfg = resolve_selection_profile(
            cfg,
            profile=str(args.selection_profile),
            target_final_k=args.target_final_k,
        )
        cfg.selection.metric_backend = str(args.metric_backend)
        cfg.selection.use_total_energy = bool(str(args.energy_semantics).strip().lower() == "total")
        cfg.selection.structure_metric_threshold = float(args.structure_metric_threshold)
        cfg.selection.pair_energy_gap_ev = float(args.pair_energy_gap_ev)
    elif args.target_final_k is not None:
        cfg.selection.target_final_k = int(args.target_final_k)
    return cfg


def write_rows(rows: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "benchmark_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (out_root / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            "# Conformer Backend Benchmark",
            "",
            "| backend | status | n_raw_frames | n_preselected | n_after_loose_filter | n_selected | energy_min_ev | energy_mean_ev | walltime_generation_s | walltime_total_s | raw_energy_source | generator_graph_source | work_dir |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("backend", "")),
                        str(row.get("status", "")),
                        str(row.get("n_raw_frames", "")),
                        str(row.get("n_preselected", "")),
                        str(row.get("n_after_loose_filter", "")),
                        str(row.get("n_selected", "")),
                        str(row.get("energy_min_ev", "")),
                        str(row.get("energy_mean_ev", "")),
                        str(row.get("walltime_generation_s", "")),
                        str(row.get("walltime_total_s", "")),
                        str(row.get("raw_energy_source", "")),
                        str(row.get("generator_graph_source", "")),
                        str(row.get("work_dir", "")),
                    ]
                )
                + " |"
            )
        (out_root / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    mol = read_molecule_any(input_path)
    out_root = Path(args.out_root)
    rows: list[dict] = []
    for backend in [x.strip() for x in str(args.backends).split(",") if x.strip()]:
        job_name = f"{input_path.stem}__{backend}"
        cfg = make_config(args, backend=backend, out_root=out_root / backend)
        sampler = ConformerMDSampler(config=cfg)
        t0 = time.perf_counter()
        try:
            result = sampler.run(molecule=mol.copy(), job_name=job_name)
            elapsed = time.perf_counter() - t0
            summary = dict(result.metadata.get("result_summary", {}))
            gen_summary = dict(result.metadata.get("generator_summary", {}))
            rows.append(
                {
                    "backend": str(backend),
                    "status": "ok",
                    "job_name": str(job_name),
                    "n_raw_frames": int(result.metadata.get("n_raw_frames", 0)),
                    "n_preselected": int(result.metadata.get("n_preselected", 0)),
                    "n_after_loose_filter": int(result.metadata.get("n_after_loose_filter", 0)),
                    "n_selected": int(result.metadata.get("n_selected", 0)),
                    "energy_min_ev": summary.get("energy_min_ev"),
                    "energy_mean_ev": summary.get("energy_mean_ev"),
                    "walltime_generation_s": gen_summary.get("walltime_generation_s"),
                    "walltime_total_s": float(elapsed),
                    "raw_energy_source": result.metadata.get("raw_energy_source"),
                    "generator_graph_source": gen_summary.get("graph_source"),
                    "work_dir": str(cfg.output.work_dir / job_name),
                    "error": "",
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "backend": str(backend),
                    "status": "failed",
                    "job_name": str(job_name),
                    "n_raw_frames": 0,
                    "n_preselected": 0,
                    "n_after_loose_filter": 0,
                    "n_selected": 0,
                    "energy_min_ev": None,
                    "energy_mean_ev": None,
                    "walltime_generation_s": None,
                    "walltime_total_s": float(elapsed),
                    "raw_energy_source": "",
                    "generator_graph_source": "",
                    "work_dir": str(cfg.output.work_dir / job_name),
                    "error": str(exc),
                }
            )
    write_rows(rows, out_root)
    print((out_root / "benchmark_summary.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
