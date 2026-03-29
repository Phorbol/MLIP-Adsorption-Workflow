from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, GeometryPairDistanceDescriptor, read_molecule_any


def _set_nested(d: dict, key: str, value):
    keys = key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def _build_cfg(root: Path, cfgd: dict, model_path: str) -> ConformerMDSamplerConfig:
    cfg = ConformerMDSamplerConfig()
    cfg.output.work_dir = root / "artifacts" / "conformer_md" / "report_runs_20260314"
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


def _to_list(frames):
    if isinstance(frames, list):
        return frames
    return [frames]


def _project_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(2, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    k = min(2, u.shape[1])
    coords = u[:, :k] * s[:k]
    if k < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - k)))
    ev = s * s
    ratio = ev / np.sum(ev) if np.sum(ev) > 0 else np.zeros_like(ev)
    out_ratio = np.zeros(2, dtype=float)
    out_ratio[: min(2, len(ratio))] = ratio[:2]
    return coords, out_ratio


def _plot_stage_pca(case_dir: Path, stage_name: str, stage_file: str):
    f = case_dir / stage_file
    if not f.exists():
        return
    frames = _to_list(read(f.as_posix(), index=":"))
    if len(frames) < 2:
        return
    feats = GeometryPairDistanceDescriptor().transform(frames)
    xy, ratio = _project_2d(feats)
    energies = np.asarray([a.info.get("energy_ev", np.nan) for a in frames], dtype=float)
    has_energy = bool(np.any(np.isfinite(energies)))
    plt.figure(figsize=(5.2, 4.2), dpi=140)
    if has_energy:
        mask = np.isfinite(energies)
        plt.scatter(xy[mask, 0], xy[mask, 1], c=energies[mask], s=28, cmap="viridis")
        plt.colorbar(label="energy_ev")
        if np.any(~mask):
            plt.scatter(xy[~mask, 0], xy[~mask, 1], s=20, c="gray")
    else:
        plt.scatter(xy[:, 0], xy[:, 1], s=24, c="tab:blue")
    plt.title(f"{case_dir.name} | {stage_name}")
    plt.xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
    plt.tight_layout()
    out_dir = case_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig((out_dir / f"pca_{stage_name}.png").as_posix())
    plt.close()


def _run_case(root: Path, atoms, model_path: str, name: str, base: dict, override: dict):
    cfgd = deepcopy(base)
    for k, v in override.items():
        _set_nested(cfgd, k, v)
    cfg = _build_cfg(root, cfgd, model_path)
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
    case_dir = cfg.output.work_dir / name
    stages = [
        ("preselected", "preselected.extxyz"),
        ("loose_relaxed", "loose_relaxed.extxyz"),
        ("loose_filtered", "loose_filtered.extxyz"),
        ("refined", "refined.extxyz"),
        ("final", "ensemble.extxyz"),
    ]
    for sname, sfile in stages:
        _plot_stage_pca(case_dir, sname, sfile)
    return row


def _render_report(report_dir: Path, rows: list[dict]):
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    lines = []
    lines.append("# Conformer 多阶段采样-松弛实验报告")
    lines.append("")
    lines.append("## 指标解释")
    lines.append("")
    lines.append("- final_diversity_pair_mean：最终系综中，基于几何对距离描述符的两两欧氏距离均值。值越大表示构象分布越分散、覆盖更广；值越小表示构象更集中。")
    lines.append("- final_energy_mean / final_energy_min：最终系综能量均值/最小值，越低通常代表更稳定构象更充分。")
    lines.append("- pre_to_loose_rms_mean / loose_to_refine_rms_mean：两次松弛前后平均RMS位移，反映松弛“力度”。")
    lines.append("")
    lines.append("## 扫描总览")
    lines.append("")
    lines.append("| case | n_final | final_over_raw | E_mean | E_min | diversity | loose_removed | final_removed | pre->loose RMS | loose->refine RMS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['case']} | {fmt(r['n_final'])} | {fmt(r['final_over_raw'])} | {fmt(r['final_energy_mean'])} | {fmt(r['final_energy_min'])} | {fmt(r['final_diversity_pair_mean'])} | {fmt(r['loose_removed'])} | {fmt(r['final_removed'])} | {fmt(r['pre_to_loose_rms_mean'])} | {fmt(r['loose_to_refine_rms_mean'])} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    best_e = min(rows, key=lambda x: x["final_energy_mean"])
    best_div = max(rows, key=lambda x: x["final_diversity_pair_mean"])
    lines.append(f"- 能量最优：**{best_e['case']}**（final_energy_mean={best_e['final_energy_mean']:.6f}）。")
    lines.append(f"- 多样性最高：**{best_div['case']}**（final_diversity_pair_mean={best_div['final_diversity_pair_mean']:.6f}）。")
    lines.append("- FPS 与降维聚类差异：在相同 relax 与筛选参数下，对比 `fps_baseline` 与 `fps_pca_kmeans_baseline`。")
    lines.append("- long relax 差异：对比 `fps_long_relax` / `fps_pca_kmeans_long_relax`，观察更大 refine 步数对能量和多样性的再平衡。")
    lines.append("")
    lines.append("## PCA 图")
    lines.append("")
    for r in rows:
        case = r["case"]
        lines.append(f"### {case}")
        lines.append("")
        for stage in ["preselected", "loose_relaxed", "loose_filtered", "refined", "final"]:
            p = f"./{case}/plots/pca_{stage}.png"
            lines.append(f"![{case}-{stage}]({p})")
            lines.append("")
    (report_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "artifacts" / "conformer_md" / "report_runs_20260314"
    report_dir.mkdir(parents=True, exist_ok=True)
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
        "loose": {"maxf": 0.5, "steps": 50},
        "refine": {"maxf": 0.05, "steps": 300},
    }
    cases = [
        ("fps_baseline", {"selection.mode": "fps", "loose.steps": 25, "refine.steps": 50}),
        ("fps_pca_kmeans_baseline", {"selection.mode": "fps_pca_kmeans", "loose.steps": 25, "refine.steps": 50}),
        ("fps_long_relax", {"selection.mode": "fps", "loose.steps": 50, "refine.steps": 300}),
        ("fps_pca_kmeans_long_relax", {"selection.mode": "fps_pca_kmeans", "loose.steps": 50, "refine.steps": 300}),
        ("fps_pca_kmeans_long_tight", {"selection.mode": "fps_pca_kmeans", "loose.steps": 50, "refine.steps": 300, "selection.energy_window_ev": 0.15, "selection.rmsd_threshold": 0.04}),
    ]
    rows = []
    for name, override in cases:
        print(f"=== RUN {name} ===")
        row = _run_case(root=root, atoms=atoms, model_path=model_path, name=name, base=base, override=override)
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)
    (report_dir / "report_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_report(report_dir=report_dir, rows=rows)
    print(f"SAVED {(report_dir / 'final_report.md').as_posix()}")


if __name__ == "__main__":
    main()
