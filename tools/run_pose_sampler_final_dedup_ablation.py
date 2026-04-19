from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ase.build import fcc111, molecule
from ase.io import read

from adsorption_ensemble.basin import BasinConfig, run_named_basin_ablation
from adsorption_ensemble.relax.backends import IdentityRelaxBackend


def _case_registry(base_root: Path) -> dict[str, dict]:
    return {
        "Cu_fcc111__CO": {
            "root": base_root / "Cu_fcc111" / "CO",
            "slab": fcc111("Cu", size=(4, 4, 4), vacuum=12.0),
            "adsorbate": molecule("CO"),
        },
        "Cu_fcc111__H2O": {
            "root": base_root / "Cu_fcc111" / "H2O",
            "slab": fcc111("Cu", size=(4, 4, 4), vacuum=12.0),
            "adsorbate": molecule("H2O"),
        },
        "Pt_fcc111__NH3": {
            "root": base_root / "Pt_fcc111" / "NH3",
            "slab": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
            "adsorbate": molecule("NH3"),
        },
    }


def _configs() -> dict[str, BasinConfig]:
    base = BasinConfig(
        energy_window_ev=2.5,
        desorption_min_bonds=1,
        binding_tau=1.15,
        signature_mode="provenance",
        final_basin_merge_metric="off",
        work_dir=None,
    )
    return {
        "binding_surface": replace(
            base,
            dedup_metric="binding_surface_distance",
            dedup_cluster_method="greedy",
            surface_descriptor_threshold=0.30,
            surface_descriptor_nearest_k=8,
            surface_descriptor_atom_mode="binding_only",
            surface_descriptor_relative=False,
            surface_descriptor_rmsd_gate=0.25,
        ),
        "signature_only": replace(base, dedup_metric="signature_only"),
        "rmsd_0p10": replace(base, dedup_metric="rmsd", dedup_cluster_method="hierarchical", rmsd_threshold=0.10),
        "mace_node_l2_0p20": replace(
            base,
            dedup_metric="mace_node_l2",
            dedup_cluster_method="hierarchical",
            mace_node_l2_threshold=0.20,
            mace_model_path="/root/.cache/mace/mace-omat-0-small.model",
            mace_device="cuda",
            mace_dtype="float32",
            mace_head_name="omat_pbe",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run three-way final-dedup ablation on pose sampler audit cases.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("artifacts/autoresearch/pose_sampler_audit_20260405"),
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["Cu_fcc111__CO", "Cu_fcc111__H2O", "Pt_fcc111__NH3"],
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("artifacts/autoresearch/pose_sampler_audit_20260405/final_dedup_ablation"),
    )
    args = parser.parse_args()

    registry = _case_registry(Path(args.benchmark_root))
    configs = _configs()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_name in args.cases:
        if case_name not in registry:
            raise ValueError(f"Unknown case: {case_name}")
        case = registry[case_name]
        frames = list(read((case["root"] / "basin_work" / "relax" / "relaxed_stream.extxyz").as_posix(), index=":"))
        result = run_named_basin_ablation(
            frames=frames,
            slab_ref=case["slab"],
            adsorbate_ref=case["adsorbate"],
            slab_n=len(case["slab"]),
            normal_axis=2,
            configs=configs,
            relax_backend=IdentityRelaxBackend(),
        )
        slim = {
            "case": str(case_name),
            "n_relaxed_frames": int(len(frames)),
            "comparison": dict(result.get("comparison", {})),
            "configs": {
                str(name): {
                    "status": str(payload["status"]),
                    "n_basins": int(payload["n_basins"]),
                    "n_rejected": int(payload["n_rejected"]),
                }
                for name, payload in result["configs"].items()
            },
        }
        rows.append(slim)
        (out_root / f"{case_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
