from __future__ import annotations

import json
import traceback
from copy import deepcopy
from pathlib import Path

from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, read_molecule_any


def _set_nested(d: dict, key: str, value):
    keys = key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def _build_config(root: Path, cfgd: dict, model_path: str) -> ConformerMDSamplerConfig:
    cfg = ConformerMDSamplerConfig()
    cfg.output.work_dir = root / "artifacts" / "conformer_md" / "sweep_runs_20260314"
    cfg.output.save_all_frames = False
    cfg.md.n_runs = int(cfgd["md"]["n_runs"])
    cfg.md.temperature_k = float(cfgd["md"]["temperature_k"])
    cfg.md.time_ps = float(cfgd["md"]["time_ps"])
    cfg.md.step_fs = float(cfgd["md"]["step_fs"])
    cfg.md.dump_fs = float(cfgd["md"]["dump_fs"])
    cfg.selection.mode = cfgd["selection"]["mode"]
    cfg.selection.preselect_k = int(cfgd["selection"]["preselect_k"])
    cfg.selection.pca_variance_threshold = float(cfgd["selection"]["pca_variance_threshold"])
    cfg.selection.fps_pool_factor = int(cfgd["selection"]["fps_pool_factor"])
    cfg.selection.energy_window_ev = float(cfgd["selection"]["energy_window_ev"])
    cfg.selection.rmsd_threshold = float(cfgd["selection"]["rmsd_threshold"])
    cfg.descriptor.backend = "mace"
    cfg.relax.backend = "mace_relax"
    cfg.descriptor.mace.model_path = model_path
    cfg.relax.mace.model_path = model_path
    cfg.descriptor.mace.head_name = "omol"
    cfg.relax.mace.head_name = "omol"
    cfg.descriptor.mace.device = "cuda"
    cfg.relax.mace.device = "cuda"
    cfg.descriptor.mace.dtype = "float32"
    cfg.relax.mace.dtype = "float32"
    cfg.descriptor.mace.max_edges_per_batch = 15000
    cfg.relax.mace.max_edges_per_batch = 15000
    cfg.relax.loose.maxf = float(cfgd["loose"]["maxf"])
    cfg.relax.loose.steps = int(cfgd["loose"]["steps"])
    cfg.relax.refine.maxf = float(cfgd["refine"]["maxf"])
    cfg.relax.refine.steps = int(cfgd["refine"]["steps"])
    return cfg


def main():
    root = Path(__file__).resolve().parents[1]
    work_root = root / "artifacts" / "conformer_md" / "sweep_runs_20260314"
    work_root.mkdir(parents=True, exist_ok=True)
    atoms = read_molecule_any(root / "C6.gjf")
    model_path = "/root/.cache/mace/mace-mh-1.model"
    base = {
        "md": {"n_runs": 1, "temperature_k": 400.0, "time_ps": 8.0, "step_fs": 1.0, "dump_fs": 8.0},
        "selection": {
            "mode": "fps_pca_kmeans",
            "preselect_k": 32,
            "pca_variance_threshold": 0.95,
            "fps_pool_factor": 3,
            "energy_window_ev": 0.20,
            "rmsd_threshold": 0.05,
        },
        "loose": {"maxf": 0.5, "steps": 25},
        "refine": {"maxf": 0.05, "steps": 50},
    }
    cases = [
        ("baseline_fast", {}),
        ("low_preselect", {"selection.preselect_k": 16}),
        ("loose_faster", {"loose.maxf": 0.8, "loose.steps": 15}),
        ("refine_tight", {"refine.maxf": 0.03, "refine.steps": 80}),
    ]
    results = []
    for name, override in cases:
        cfgd = deepcopy(base)
        for k, v in override.items():
            _set_nested(cfgd, k, v)
        cfg = _build_config(root=root, cfgd=cfgd, model_path=model_path)
        print(f"=== RUN {name} ===")
        try:
            sampler = ConformerMDSampler(config=cfg)
            out = sampler.run(atoms, job_name=name)
            m = out.metadata["stage_metrics"]
            row = {
                "case": name,
                "ok": True,
                "n_final": m["counts"]["final"],
                "final_over_raw": m["retention"]["final_over_raw"],
                "final_energy_mean": (m["energy"]["final"] or {}).get("mean"),
                "final_energy_min": (m["energy"]["final"] or {}).get("min"),
                "final_diversity_pair_mean": (m["diversity"]["final"] or {}).get("pair_mean"),
                "loose_removed": m["dedup_removed"]["loose_filter_removed"],
                "final_removed": m["dedup_removed"]["final_filter_removed"],
                "pre_to_loose_rms_mean": (m["relax_shift"]["pre_to_loose"] or {}).get("mean"),
                "loose_to_refine_rms_mean": (m["relax_shift"]["loose_filtered_to_refined"] or {}).get("mean"),
                "config": cfgd,
            }
            print(json.dumps(row, ensure_ascii=False))
            results.append(row)
        except Exception as e:
            traceback.print_exc()
            results.append({"case": name, "ok": False, "error": str(e), "config": cfgd})
    (work_root / "sweep_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_rows = [
        "case,ok,n_final,final_over_raw,final_energy_mean,final_energy_min,final_diversity_pair_mean,loose_removed,final_removed,pre_to_loose_rms_mean,loose_to_refine_rms_mean"
    ]
    for r in results:
        csv_rows.append(
            ",".join(
                [
                    str(r.get("case", "")),
                    str(r.get("ok", "")),
                    str(r.get("n_final", "")),
                    str(r.get("final_over_raw", "")),
                    str(r.get("final_energy_mean", "")),
                    str(r.get("final_energy_min", "")),
                    str(r.get("final_diversity_pair_mean", "")),
                    str(r.get("loose_removed", "")),
                    str(r.get("final_removed", "")),
                    str(r.get("pre_to_loose_rms_mean", "")),
                    str(r.get("loose_to_refine_rms_mean", "")),
                ]
            )
        )
    (work_root / "sweep_summary.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(f"SAVED {(work_root / 'sweep_summary.json').as_posix()}")


if __name__ == "__main__":
    main()
