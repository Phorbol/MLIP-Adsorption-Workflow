from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, read_molecule_any
from adsorption_ensemble.conformer_md.config import MACEInferenceConfig
from adsorption_ensemble.conformer_md.mace_inference import MACEBatchInferencer
from adsorption_ensemble.conformer_md.descriptors import GeometryPairDistanceDescriptor
from adsorption_ensemble.conformer_md.xtb import XTBMDConfig, XTBMDRunner


def _set_nested(d: dict, key: str, value):
    keys = key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


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


def _to_list(frames):
    if isinstance(frames, list):
        return frames
    return [frames]


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
    out_dir = case_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
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
    plt.savefig((out_dir / f"pca_{stage_name}.png").as_posix())
    plt.close()


def _render_summary_plots(report_dir: Path, rows: list[dict]):
    out_dir = report_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = np.asarray([r["final_diversity_pair_mean"] for r in rows], dtype=float)
    ym = np.asarray([r["final_energy_mean"] for r in rows], dtype=float)
    y0 = np.asarray([r["final_energy_min"] for r in rows], dtype=float)
    n = np.asarray([r["n_final"] for r in rows], dtype=float)
    plt.figure(figsize=(6.4, 4.6), dpi=150)
    plt.scatter(xs, ym, s=20 + 8 * np.sqrt(np.maximum(n, 0.0)), c=n, cmap="plasma")
    for r in rows:
        plt.annotate(r["case"], (r["final_diversity_pair_mean"], r["final_energy_mean"]), fontsize=7, alpha=0.9)
    plt.colorbar(label="n_final")
    plt.xlabel("final_diversity_pair_mean")
    plt.ylabel("final_energy_mean (eV/atom)")
    plt.title("Energy–Diversity Tradeoff (Mean)")
    plt.tight_layout()
    plt.savefig((out_dir / "tradeoff_energy_mean_vs_diversity.png").as_posix())
    plt.close()
    plt.figure(figsize=(6.4, 4.6), dpi=150)
    plt.scatter(xs, y0, s=20 + 8 * np.sqrt(np.maximum(n, 0.0)), c=n, cmap="viridis")
    for r in rows:
        plt.annotate(r["case"], (r["final_diversity_pair_mean"], r["final_energy_min"]), fontsize=7, alpha=0.9)
    plt.colorbar(label="n_final")
    plt.xlabel("final_diversity_pair_mean")
    plt.ylabel("final_energy_min (eV/atom)")
    plt.title("Energy–Diversity Tradeoff (Min)")
    plt.tight_layout()
    plt.savefig((out_dir / "tradeoff_energy_min_vs_diversity.png").as_posix())
    plt.close()


def _render_report(report_dir: Path, md_info: dict, cases: list[dict], rows: list[dict]):
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    best_e = min(rows, key=lambda x: x["final_energy_mean"])
    best_div = max(rows, key=lambda x: x["final_diversity_pair_mean"])
    best_size = max(rows, key=lambda x: x["n_final"])
    lines: list[str] = []
    lines.append("# Conformer 多阶段采样-松弛最终实验报告（600K 100ps 单次MD复用）")
    lines.append("")
    lines.append("## 输入与一次性计算")
    lines.append("")
    lines.append(f"- input: `{md_info['input']}`")
    lines.append(f"- MD: temperature_k={md_info['md']['temperature_k']}, time_ps={md_info['md']['time_ps']}, step_fs={md_info['md']['step_fs']}, dump_fs={md_info['md']['dump_fs']}, n_runs={md_info['md']['n_runs']}")
    lines.append(f"- n_raw_frames: {md_info['n_raw_frames']}")
    lines.append(f"- raw_frames_file: `{md_info['raw_frames_file']}`")
    lines.append(f"- mace_raw_features_file: `{md_info['mace_raw_features_file']}`")
    lines.append("")
    lines.append("## 可调参数清单（本次覆盖的核心）")
    lines.append("")
    lines.append("- 采样：selection.mode（fps / kmeans / fps_pca_kmeans），preselect_k，pca_variance_threshold，fps_pool_factor")
    lines.append("- 两次去重/筛选：loose_filter / final_filter（none/energy/rmsd/dual），energy_window_ev，rmsd_threshold，loose_* 与 final_* 阈值覆盖")
    lines.append("- 两段松弛：loose(maxf,steps) 与 refine(maxf,steps)")
    lines.append("- MACE：model_path/head_name/device/dtype/max_edges_per_batch（本次MACE特征对原始帧只算一次）")
    lines.append("")
    lines.append("## 实验矩阵")
    lines.append("")
    lines.append("| case | mode | preK | pca95 | loose_steps | refine_steps | loose_filter | final_filter | dE | rmsd |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---:|---:|")
    for c in cases:
        lines.append(
            f"| {c['name']} | {c['cfg']['selection']['mode']} | {c['cfg']['selection']['preselect_k']} | {c['cfg']['selection']['pca_variance_threshold']} | {c['cfg']['loose']['steps']} | {c['cfg']['refine']['steps']} | {c['cfg']['selection']['loose_filter']} | {c['cfg']['selection']['final_filter']} | {c['cfg']['selection']['energy_window_ev']} | {c['cfg']['selection']['rmsd_threshold']} |"
        )
    lines.append("")
    lines.append("## 结果总览（核心指标）")
    lines.append("")
    lines.append("| case | n_final | final_over_raw | E_mean | E_min | diversity | loose_removed | final_removed | pre->loose RMS | loose->refine RMS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['case']} | {fmt(r['n_final'])} | {fmt(r['final_over_raw'])} | {fmt(r['final_energy_mean'])} | {fmt(r['final_energy_min'])} | {fmt(r['final_diversity_pair_mean'])} | {fmt(r['loose_removed'])} | {fmt(r['final_removed'])} | {fmt(r['pre_to_loose_rms_mean'])} | {fmt(r['loose_to_refine_rms_mean'])} |"
        )
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    lines.append("![tradeoff-mean](./plots/tradeoff_energy_mean_vs_diversity.png)")
    lines.append("")
    lines.append("![tradeoff-min](./plots/tradeoff_energy_min_vs_diversity.png)")
    lines.append("")
    lines.append("## 推荐")
    lines.append("")
    lines.append(f"- 追求最低能（平均）：优先 **{best_e['case']}**（E_mean={best_e['final_energy_mean']:.6f}，n_final={best_e['n_final']}）")
    lines.append(f"- 追求最大覆盖（diversity）：优先 **{best_div['case']}**（diversity={best_div['final_diversity_pair_mean']:.6f}，n_final={best_div['n_final']}）")
    lines.append(f"- 追求更大系综规模（n_final）：优先 **{best_size['case']}**（n_final={best_size['n_final']}，E_mean={best_size['final_energy_mean']:.6f}，diversity={best_size['final_diversity_pair_mean']:.6f}）")
    lines.append("- 默认推荐（平衡）：选择在 tradeoff 图上靠近左下且 n_final 不过小的点，通常是 fps_pca_kmeans + long refine + dual 终筛 的组合。")
    lines.append("")
    lines.append("## 分阶段PCA投影图（每个case）")
    lines.append("")
    for r in rows:
        case = r["case"]
        lines.append(f"### {case}")
        lines.append("")
        for stage in ["preselected", "loose_relaxed", "loose_filtered", "refined", "final"]:
            lines.append(f"![{case}-{stage}](./{case}/plots/pca_{stage}.png)")
            lines.append("")
    (report_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    input_gjf = root / "C6.gjf"
    atoms = read_molecule_any(input_gjf)
    report_dir = root / "artifacts" / "conformer_md" / "final_report_600K_100ps"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_dir = report_dir / "md_run"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_cfg = XTBMDConfig(temperature_k=600.0, time_ps=100.0, step_fs=1.0, dump_fs=50.0, n_runs=1)
    raw_frames_file = md_dir / "all_frames.extxyz"
    md_meta_file = md_dir / "md_metadata.json"
    if raw_frames_file.exists() and md_meta_file.exists():
        frames = _to_list(read(raw_frames_file.as_posix(), index=":"))
        md_metadata = json.loads(md_meta_file.read_text(encoding="utf-8"))
    else:
        md_res = XTBMDRunner(md_cfg).run(atoms, md_dir)
        frames = md_res.frames
        write(raw_frames_file.as_posix(), frames)
        md_meta_file.write_text(json.dumps(md_res.metadata, indent=2, default=str), encoding="utf-8")
        md_metadata = dict(md_res.metadata)
    infer_cfg = MACEInferenceConfig(
        model_path="/root/.cache/mace/mace-mh-1.model",
        device="cuda",
        dtype="float32",
        max_edges_per_batch=15000,
        num_workers=1,
        layers_to_keep=-1,
        mlp_energy_key=None,
        head_name="omol",
    )
    mace_raw_features_file = md_dir / "mace_raw_features.npy"
    mace_raw_features_meta_file = md_dir / "mace_raw_features_meta.json"
    if mace_raw_features_file.exists() and mace_raw_features_meta_file.exists():
        raw_features = np.load(mace_raw_features_file.as_posix())
        mace_raw_meta = json.loads(mace_raw_features_meta_file.read_text(encoding="utf-8"))
    else:
        infer = MACEBatchInferencer(infer_cfg)
        out = infer.infer(frames)
        raw_features = np.asarray(out.descriptors, dtype=float)
        np.save(mace_raw_features_file.as_posix(), raw_features)
        mace_raw_features_meta_file.write_text(json.dumps(out.metadata, indent=2, default=str), encoding="utf-8")
        mace_raw_meta = dict(out.metadata)
    base = {
        "selection": {
            "mode": "fps_pca_kmeans",
            "preselect_k": 32,
            "pca_variance_threshold": 0.95,
            "fps_pool_factor": 3,
            "energy_window_ev": 0.20,
            "rmsd_threshold": 0.05,
            "loose_filter": "dual",
            "final_filter": "dual",
            "loose_energy_window_ev": None,
            "loose_rmsd_threshold": None,
            "final_energy_window_ev": None,
            "final_rmsd_threshold": None,
        },
        "loose": {"maxf": 0.5, "steps": 50},
        "refine": {"maxf": 0.05, "steps": 300},
    }
    case_defs = [
        ("fps_dual_long", {"selection.mode": "fps", "loose.steps": 50, "refine.steps": 300, "selection.final_filter": "dual"}),
        ("fps_pca_kmeans_dual_long", {"selection.mode": "fps_pca_kmeans", "loose.steps": 50, "refine.steps": 300, "selection.final_filter": "dual"}),
        ("fps_energy_long", {"selection.mode": "fps", "loose.steps": 50, "refine.steps": 300, "selection.final_filter": "energy"}),
        ("fps_pca_kmeans_energy_long", {"selection.mode": "fps_pca_kmeans", "loose.steps": 50, "refine.steps": 300, "selection.final_filter": "energy"}),
        ("fps_dual_short", {"selection.mode": "fps", "loose.steps": 25, "refine.steps": 100, "selection.final_filter": "dual"}),
        ("fps_pca_kmeans_dual_short", {"selection.mode": "fps_pca_kmeans", "loose.steps": 25, "refine.steps": 100, "selection.final_filter": "dual"}),
        ("fps_pca_kmeans_dual_long_tight", {"selection.mode": "fps_pca_kmeans", "loose.steps": 50, "refine.steps": 300, "selection.energy_window_ev": 0.15, "selection.rmsd_threshold": 0.04}),
    ]
    cases: list[dict] = []
    for name, override in case_defs:
        cfgd = deepcopy(base)
        for k, v in override.items():
            _set_nested(cfgd, k, v)
        cases.append({"name": name, "cfg": cfgd})
    rows: list[dict] = []
    for c in cases:
        case_dir = report_dir / c["name"]
        stage_file = case_dir / "stage_metrics.json"
        if stage_file.exists():
            m = json.loads(stage_file.read_text(encoding="utf-8"))
        else:
            cfg = ConformerMDSamplerConfig()
            cfg.output.work_dir = report_dir
            cfg.output.save_all_frames = False
            cfg.descriptor.backend = "geometry"
            cfg.relax.backend = "mace_relax"
            cfg.relax.mace = infer_cfg
            cfg.selection.mode = c["cfg"]["selection"]["mode"]
            cfg.selection.preselect_k = int(c["cfg"]["selection"]["preselect_k"])
            cfg.selection.pca_variance_threshold = float(c["cfg"]["selection"]["pca_variance_threshold"])
            cfg.selection.fps_pool_factor = int(c["cfg"]["selection"]["fps_pool_factor"])
            cfg.selection.energy_window_ev = float(c["cfg"]["selection"]["energy_window_ev"])
            cfg.selection.rmsd_threshold = float(c["cfg"]["selection"]["rmsd_threshold"])
            cfg.selection.loose_filter = c["cfg"]["selection"]["loose_filter"]
            cfg.selection.final_filter = c["cfg"]["selection"]["final_filter"]
            cfg.selection.loose_energy_window_ev = c["cfg"]["selection"]["loose_energy_window_ev"]
            cfg.selection.loose_rmsd_threshold = c["cfg"]["selection"]["loose_rmsd_threshold"]
            cfg.selection.final_energy_window_ev = c["cfg"]["selection"]["final_energy_window_ev"]
            cfg.selection.final_rmsd_threshold = c["cfg"]["selection"]["final_rmsd_threshold"]
            cfg.relax.loose.maxf = float(c["cfg"]["loose"]["maxf"])
            cfg.relax.loose.steps = int(c["cfg"]["loose"]["steps"])
            cfg.relax.refine.maxf = float(c["cfg"]["refine"]["maxf"])
            cfg.relax.refine.steps = int(c["cfg"]["refine"]["steps"])
            sampler = ConformerMDSampler(config=cfg)
            out = sampler.run_from_frames(frames=frames, job_name=c["name"], md_runs_metadata=[md_metadata], raw_features=raw_features)
            m = out.metadata["stage_metrics"]
        row = {
            "case": c["name"],
            "n_final": m["counts"]["final"],
            "final_over_raw": m["retention"]["final_over_raw"],
            "final_energy_mean": (m["energy"]["final"] or {}).get("mean"),
            "final_energy_min": (m["energy"]["final"] or {}).get("min"),
            "final_diversity_pair_mean": (m["diversity"]["final"] or {}).get("pair_mean"),
            "loose_removed": m["dedup_removed"]["loose_filter_removed"],
            "final_removed": m["dedup_removed"]["final_filter_removed"],
            "pre_to_loose_rms_mean": (m["relax_shift"]["pre_to_loose"] or {}).get("mean"),
            "loose_to_refine_rms_mean": (m["relax_shift"]["loose_filtered_to_refined"] or {}).get("mean"),
        }
        rows.append(row)
        for sname, sfile in [
            ("preselected", "preselected.extxyz"),
            ("loose_relaxed", "loose_relaxed.extxyz"),
            ("loose_filtered", "loose_filtered.extxyz"),
            ("refined", "refined.extxyz"),
            ("final", "ensemble.extxyz"),
        ]:
            _plot_stage_pca(case_dir, sname, sfile)
    (report_dir / "report_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md_info = {
        "input": str(input_gjf),
        "md": {"temperature_k": md_cfg.temperature_k, "time_ps": md_cfg.time_ps, "step_fs": md_cfg.step_fs, "dump_fs": md_cfg.dump_fs, "n_runs": md_cfg.n_runs},
        "n_raw_frames": int(len(frames)),
        "raw_frames_file": str(raw_frames_file),
        "mace_raw_features_file": str(mace_raw_features_file),
        "md_metadata_file": str(md_meta_file),
        "mace_raw_features_meta_file": str(mace_raw_features_meta_file),
    }
    _render_summary_plots(report_dir=report_dir, rows=rows)
    _render_report(report_dir=report_dir, md_info=md_info, cases=cases, rows=rows)
    print((report_dir / "final_report.md").as_posix())


if __name__ == "__main__":
    main()
