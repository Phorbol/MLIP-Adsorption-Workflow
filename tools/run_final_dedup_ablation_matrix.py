from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import fcc100, fcc111, fcc211
from ase.io import read

from adsorption_ensemble.basin import BasinConfig, run_named_basin_ablation
from adsorption_ensemble.relax.backends import IdentityRelaxBackend
from autoadsorbate.Smile import atoms_from_smile
from tools.run_miller_monodentate_matrix import build_miller_metal_slab_suite, build_monodentate_suite


def _crosslib_case_registry() -> dict[str, dict[str, Any]]:
    return {
        "Pt111_methanol": {"slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CO")},
        "Pt100_methanol": {"slab": fcc100("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CO")},
        "Pt111_dimethyl_ether": {"slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("COC")},
        "Pt211_ethanol": {"slab": fcc211("Pt", size=(6, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CCO")},
        "Pt111_methylamine": {"slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0), "adsorbate": atoms_from_smile("CN")},
    }


def _miller_case_registry() -> dict[str, dict[str, Any]]:
    slabs = build_miller_metal_slab_suite()
    molecules = build_monodentate_suite()
    out: dict[str, dict[str, Any]] = {}
    for slab_name, slab in slabs.items():
        for mol_name, adsorbate in molecules.items():
            out[f"{slab_name}__{mol_name}"] = {"slab": slab, "adsorbate": adsorbate}
    return out


def _default_configs(mace_model_path: str, mace_device: str, mace_dtype: str) -> dict[str, BasinConfig]:
    common = {
        "energy_window_ev": 2.5,
        "desorption_min_bonds": 1,
        "binding_tau": 1.15,
        "mace_enable_cueq": True,
        "work_dir": None,
    }
    return {
        "prov_rmsd": BasinConfig(
            dedup_metric="rmsd",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "abs_rmsd": BasinConfig(
            dedup_metric="rmsd",
            signature_mode="absolute",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "canon_rmsd": BasinConfig(
            dedup_metric="rmsd",
            signature_mode="canonical",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "binding_surface": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="greedy",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            **common,
        ),
        "binding_surface_hierarchical": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            **common,
        ),
        "pure_rmsd": BasinConfig(
            dedup_metric="pure_rmsd",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            rmsd_threshold=0.10,
            **common,
        ),
        "pure_mace_0p05": BasinConfig(
            dedup_metric="pure_mace",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.05,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "pure_mace_0p10": BasinConfig(
            dedup_metric="pure_mace",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.10,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "abs_mace_0p05": BasinConfig(
            dedup_metric="mace_node_l2",
            signature_mode="absolute",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.05,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "abs_mace_0p10": BasinConfig(
            dedup_metric="mace_node_l2",
            signature_mode="absolute",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.10,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "abs_mace_0p20": BasinConfig(
            dedup_metric="mace_node_l2",
            signature_mode="absolute",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.20,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_canonical_mace_0p05": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="mace_node_l2",
            final_basin_merge_node_l2_threshold=0.05,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_canonical_mace_0p10": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="mace_node_l2",
            final_basin_merge_node_l2_threshold=0.10,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_pure_mace_0p05": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.05,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="none",
            final_basin_merge_use_signature_grouping=False,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_ref_canonical_mace_0p02": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.02,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="reference_canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_ref_canonical_mace_0p02_e0p05": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.02,
            final_basin_merge_energy_gate_ev=0.05,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="reference_canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_ref_canonical_mace_0p02_e0p08": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.02,
            final_basin_merge_energy_gate_ev=0.08,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="reference_canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_ref_canonical_mace_0p02_e0p10": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.02,
            final_basin_merge_energy_gate_ev=0.10,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="reference_canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_ref_canonical_mace_0p03": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.03,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="reference_canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
        "binding_surface_merge_ref_canonical_mace_0p05": BasinConfig(
            dedup_metric="binding_surface_distance",
            signature_mode="provenance",
            dedup_cluster_method="hierarchical",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
            final_basin_merge_metric="pure_mace",
            final_basin_merge_node_l2_threshold=0.05,
            final_basin_merge_cluster_method="hierarchical",
            final_basin_merge_signature_mode="reference_canonical",
            final_basin_merge_use_signature_grouping=True,
            mace_model_path=str(mace_model_path),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            **common,
        ),
    }


def _case_root(benchmark_root: Path, suite: str, case_name: str) -> Path:
    if suite == "crosslib":
        return benchmark_root / case_name / "ours"
    slab_name, ads_name = case_name.split("__", 1)
    return benchmark_root / slab_name / ads_name / "ours"


def _load_relaxed_frames(case_root: Path) -> list[Atoms]:
    path = case_root / "basin_work" / "relax" / "relaxed_stream.extxyz"
    return list(read(path.as_posix(), index=":"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    suite = str(args.suite).strip().lower()
    benchmark_root = Path(args.benchmark_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    registry = _crosslib_case_registry() if suite == "crosslib" else _miller_case_registry()
    case_names = sorted(registry.keys())
    if args.cases:
        allow = {x.strip() for x in str(args.cases).split(",") if x.strip()}
        case_names = [x for x in case_names if x in allow]
    configs = _default_configs(
        mace_model_path=str(args.mace_model_path),
        mace_device=str(args.mace_device),
        mace_dtype=str(args.mace_dtype),
    )
    if args.configs:
        allow = {x.strip() for x in str(args.configs).split(",") if x.strip()}
        configs = {name: cfg for name, cfg in configs.items() if name in allow}
        if not configs:
            raise ValueError("No configs matched --configs filter.")
    rows = []
    skipped_cases: list[dict[str, str]] = []
    for case_name in case_names:
        case_root = _case_root(benchmark_root, suite, case_name)
        relaxed_path = case_root / "basin_work" / "relax" / "relaxed_stream.extxyz"
        if not relaxed_path.exists():
            if bool(args.skip_missing):
                skipped_cases.append({"case": str(case_name), "missing_path": relaxed_path.as_posix()})
                continue
            raise FileNotFoundError(relaxed_path.as_posix())
        frames = _load_relaxed_frames(case_root)
        case = registry[case_name]
        result = run_named_basin_ablation(
            frames=frames,
            slab_ref=case["slab"],
            adsorbate_ref=case["adsorbate"],
            slab_n=len(case["slab"]),
            normal_axis=2,
            configs=configs,
            relax_backend=IdentityRelaxBackend(),
        )
        row = {
            "case": case_name,
            "suite": suite,
            "case_root": case_root.as_posix(),
            "n_relaxed_frames": int(len(frames)),
            "configs": result["configs"],
            "comparison": result.get("comparison", {}),
        }
        rows.append(row)
        _write_json(out_root / f"{case_name}.json", row)
    payload = {
        "suite": suite,
        "benchmark_root": benchmark_root.as_posix(),
        "config_names": list(configs.keys()),
        "skipped_cases": skipped_cases,
        "rows": rows,
    }
    out_path = out_root / "final_dedup_ablation_matrix.json"
    _write_json(out_path, payload)
    md_lines = [
        "# Final Dedup Ablation Matrix",
        "",
        f"- Suite: {suite}",
        f"- Cases: {len(rows)}",
        "",
    ]
    for row in rows:
        md_lines.append(f"## {row['case']}")
        md_lines.append("")
        md_lines.append(f"- Relaxed frames: {row['n_relaxed_frames']}")
        for name, cfg in row["configs"].items():
            md_lines.append(f"- {name}: status={cfg['status']}, n_basins={cfg['n_basins']}, n_rejected={cfg['n_rejected']}")
        md_lines.append("")
    md_path = out_root / "final_dedup_ablation_matrix.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(out_path.as_posix())
    print(md_path.as_posix())
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="crosslib", choices=["crosslib", "miller"])
    parser.add_argument("--benchmark-root", type=str, required=True)
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
