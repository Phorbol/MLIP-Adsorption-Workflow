from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/wsl_demo_pose")
    parser.add_argument("--model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--max-edges-per-batch", type=int, default=100000)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    strict = True
    if bool(args.strict):
        strict = True
    if bool(args.no_strict):
        strict = False

    import adsorption_ensemble.pose.sweep as sw

    sw.list_supported_molecules = lambda *a, **k: ["CO"]
    cfg = sw.PoseSweepConfig(
        max_basis_sites=12,
        postprocess_enabled=True,
        postprocess_preselect_k=64,
        postprocess_batch_relax_enabled=True,
        ensemble_enabled=True,
        ensemble_relax_backend="mace_batch",
        ensemble_relax_maxf=0.10,
        ensemble_relax_steps=80,
        ensemble_energy_window_ev=3.0,
        ensemble_dedup_metric="mace_node_l2",
        ensemble_rmsd_threshold=0.10,
        ensemble_mace_node_l2_threshold=2.0,
        ensemble_binding_tau=1.15,
        ensemble_desorption_min_bonds=1,
        node_bond_tau=1.20,
        node_hash_len=20,
        mace_model_path=str(args.model_path),
        mace_desc_device="cuda",
        mace_desc_dtype="float64",
        mace_relax_device="cuda",
        mace_relax_dtype="float32",
        mace_max_edges_per_batch=int(args.max_edges_per_batch),
        mace_head_name="",
        mace_strict=bool(strict),
        random_seed=0,
    )
    out = sw.run_pose_sampling_sweep(
        out_root=Path(args.out_root),
        cfg=cfg,
        max_molecules=1,
        max_slabs=1,
        max_atoms_per_molecule=2,
        max_combinations=1,
        slab_filter=["fcc211"],
    )
    rows = out.get("rows") or []
    ok = [r for r in rows if bool(r.get("ok"))]
    payload = {
        "run_dir": out.get("run_dir"),
        "summary_json": out.get("summary_json"),
        "n_rows": len(rows),
        "n_ok": len(ok),
    }
    if ok:
        r0 = ok[0]
        od = Path(str(r0.get("output_dir", "")))
        expected = [
            od / "pose_pool.extxyz",
            od / "pose_postprocess_metrics.json",
            od / "pose_preselect_pca.png",
            od / "pose_loose_filter_pca.png",
            od / "pose_final_filter_pca.png",
            od / "adsorption_energy_hist.png",
            od / "adsorption_energy.csv",
            od / "basins.extxyz",
            od / "basins.json",
            od / "nodes.json",
            Path(str(out.get("run_dir"))) / "fcc211" / "site_embedding_pca.png",
        ]
        payload.update(
            {
                "feature_backend": r0.get("feature_backend"),
                "postprocess_backend": r0.get("postprocess_backend"),
                "expected_files_ok": {p.name: (p.exists() and p.stat().st_size > 0) for p in expected},
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
