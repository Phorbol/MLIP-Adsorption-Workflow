from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from math import gcd

import numpy as np
from ase import Atom
from ase.build import bcc100, bcc110, bcc111, bulk, fcc100, fcc110, fcc111, fcc211, surface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.site import (
    PrimitiveBuilder,
    PrimitiveEmbedder,
    PrimitiveEmbeddingConfig,
    build_site_dictionary,
    compare_graph_vs_delaunay,
    enumerate_primitives_delaunay,
)
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector, export_surface_detection_report
from adsorption_ensemble.visualization import plot_inequivalent_sites_2d, plot_site_centers_only, plot_surface_primitives_2d


def build_cases():
    cu_bulk = bulk("Cu", "fcc", a=3.6, cubic=True)
    mgo_bulk = bulk("MgO", "rocksalt", a=4.21, cubic=True)
    base: dict[str, object] = {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
        "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
        "fcc110": fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
        "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
        "bcc100": bcc100("Fe", size=(4, 4, 4), vacuum=12.0),
        "bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=12.0),
        "bcc111": bcc111("Fe", size=(4, 4, 4), vacuum=12.0),
    }

    base["fcc111_s3_l5"] = fcc111("Pt", size=(3, 3, 5), vacuum=12.0)
    base["fcc110_s5_l6"] = fcc110("Pt", size=(5, 4, 6), vacuum=12.0)
    base["fcc100_s5_l3"] = fcc100("Pt", size=(5, 5, 3), vacuum=12.0)
    base["fcc211_s9_l5"] = fcc211("Pt", size=(9, 4, 5), vacuum=12.0)
    base["bcc110_s5_l6"] = bcc110("Fe", size=(5, 4, 6), vacuum=12.0)
    base["bcc100_s3_l5"] = bcc100("Fe", size=(3, 3, 5), vacuum=12.0)

    hkls = []
    for h in range(1, 5):
        for k in range(0, h + 1):
            for l in range(0, k + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                g = gcd(gcd(h, k), l) if l != 0 else gcd(h, k)
                if g > 1:
                    continue
                hkls.append((h, k, l))
    selected_hkls = [hkl for hkl in hkls if hkl in {(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1), (3, 2, 1)}]
    for hkl in selected_hkls:
        for layers in (3, 4, 6):
            key = f"surface_cu_{hkl[0]}{hkl[1]}{hkl[2]}_l{layers}"
            base[key] = surface(cu_bulk, hkl, layers=layers, vacuum=12.0)

    for hkl in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
        for layers in (4, 6):
            key = f"mgo_{hkl[0]}{hkl[1]}{hkl[2]}_l{layers}"
            slab = surface(mgo_bulk, hkl, layers=layers, vacuum=12.0).repeat((2, 2, 1))
            base[key] = slab

    for hkl in ((1, 1, 1), (2, 1, 1), (1, 0, 0)):
        for layers in (4, 6):
            key = f"alloy_cuni_{hkl[0]}{hkl[1]}{hkl[2]}_l{layers}"
            slab = surface(cu_bulk, hkl, layers=layers, vacuum=12.0).repeat((2, 2, 1))
            base[key] = make_alloy_from_top(slab, dopant="Ni", depth=2.8, every=3)

    out = dict(base)
    defect_keys = [
        "fcc111",
        "fcc110",
        "fcc211",
        "mgo_100_l4",
        "mgo_111_l4",
        "alloy_cuni_111_l4",
        "alloy_cuni_211_l4",
    ]
    for key in defect_keys:
        if key not in base:
            continue
        out[f"{key}_vacancy"] = make_vacancy_defect(base[key])
        out[f"{key}_adatom"] = make_adatom_defect(base[key])
    return out


def make_vacancy_defect(slab, normal_axis: int = 2):
    s = slab.copy()
    z = s.get_positions()[:, normal_axis]
    idx = int(np.argmax(z))
    del s[idx]
    return s


def make_adatom_defect(slab, normal_axis: int = 2):
    s = slab.copy()
    z = s.get_positions()[:, normal_axis]
    zmax = float(np.max(z))
    top_ids = np.where(z > zmax - 0.2)[0]
    center = np.mean(s.get_positions()[top_ids], axis=0)
    pos = center.copy()
    pos[normal_axis] += 1.8
    top_symbol = s[int(np.argmax(z))].symbol
    s.append(Atom(top_symbol, position=pos))
    return s


def make_alloy_from_top(slab, dopant: str, depth: float, every: int, normal_axis: int = 2):
    s = slab.copy()
    z = s.get_positions()[:, normal_axis]
    zmax = float(np.max(z))
    cand = [int(i) for i in np.where(z > zmax - depth)[0]]
    cand = sorted(cand, key=lambda i: (z[i], i), reverse=True)
    symbols = s.get_chemical_symbols()
    for ii, atom_id in enumerate(cand):
        if ii % every == 0:
            symbols[atom_id] = dopant
    s.set_chemical_symbols(symbols)
    return s


def min_surface_center_distance(primitives, slab, surface_ids):
    if len(surface_ids) == 0:
        return np.nan
    s = slab.get_positions()[surface_ids]
    builder = PrimitiveBuilder(min_site_distance=0.1)
    dmin = np.inf
    for p in primitives:
        if p.kind not in {"3c", "4c"}:
            continue
        d = builder._point_to_surface_min_distance_mic(slab, p.center, surface_ids)
        dmin = min(dmin, d)
    return float(dmin if np.isfinite(dmin) else np.nan)


def z_level_stats(slab, surface_ids, normal_axis):
    if len(surface_ids) == 0:
        return np.nan, np.nan, np.nan, 0
    z = slab.get_positions()[surface_ids, normal_axis]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    span = zmax - zmin
    levels = []
    for val in sorted(z):
        if not levels or abs(val - levels[-1]) > 0.35:
            levels.append(float(val))
    return zmin, zmax, span, len(levels)


def target_surface_count(n_atoms: int, fraction: float = 0.25):
    if n_atoms % 4 == 0:
        return int(round(n_atoms * fraction))
    if n_atoms % 4 == 1:
        return int(np.ceil(n_atoms * fraction))
    if n_atoms % 4 == 3:
        return int(np.floor(n_atoms * fraction))
    return int(round(n_atoms * fraction))


def case_family(case_name: str) -> str:
    if case_name.startswith("alloy_"):
        return "alloy"
    if case_name.startswith("mgo_"):
        return "oxide"
    if case_name.startswith("surface_cu_"):
        return "high_index_metal"
    if case_name.startswith("bcc"):
        return "bcc_metal"
    if case_name.startswith("fcc"):
        return "fcc_metal"
    return "other"


def make_surface_atom_features(slab, model: str = "small", device: str = "cpu"):
    backend = "mace_mp"
    try:
        from mace.calculators import mace_mp

        if not hasattr(make_surface_atom_features, "_calc_cache"):
            make_surface_atom_features._calc_cache = {}
        key = f"{model}|{device}"
        if key not in make_surface_atom_features._calc_cache:
            make_surface_atom_features._calc_cache[key] = mace_mp(model=model, device=device, default_dtype="float64")
        calc = make_surface_atom_features._calc_cache[key]
        feats = calc.get_descriptors(slab)
        return np.asarray(feats, dtype=float), backend
    except Exception:
        backend = "atomic_number_fallback"
        z = slab.get_atomic_numbers().astype(float)
        z = z / (np.max(z) + 1e-12)
        feats = z.reshape(-1, 1)
        return feats, backend


def main():
    out_root = Path("artifacts") / "sweep"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "surface_sweep_summary.csv"
    rows = []
    run_stamp = datetime.now().isoformat(timespec="seconds")
    report_lines = ["# Surface Sweep Report", "", f"- run: {run_stamp}", f"- run_dir: {out_dir.as_posix()}", ""]
    cases = build_cases()
    settings = [
        {"grid_step": 0.45, "spacing": 0.65},
        {"grid_step": 0.6, "spacing": 0.8},
        {"grid_step": 0.8, "spacing": 1.0},
        {"grid_step": 1.0, "spacing": 1.2},
    ]
    for case_name, slab in cases.items():
        for cfg in settings:
            pre = SurfacePreprocessor(
                min_surface_atoms=6,
                primary_detector=ProbeScanDetector(grid_step=cfg["grid_step"]),
                fallback_detector=VoxelFloodDetector(spacing=cfg["spacing"]),
                target_surface_fraction=0.25,
                target_count_mode="off",
            )
            builder = PrimitiveBuilder(min_site_distance=0.1)
            ctx = pre.build_context(slab)
            primitives = builder.build(slab, ctx)
            atom_features, feature_backend = make_surface_atom_features(slab)
            emb = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.22))
            emb_res = emb.fit_transform(slab=slab, primitives=primitives, atom_features=atom_features)
            site_dict = build_site_dictionary(emb_res.primitives, slab=slab)
            kinds = Counter(p.kind for p in primitives)
            graph_sites = {"1c": [], "2c": [], "3c": [], "4c": []}
            for p in primitives:
                graph_sites[p.kind].append(p.atom_ids)
            try:
                d_sites = enumerate_primitives_delaunay(
                    slab=slab,
                    surface_atom_ids=ctx.detection.surface_atom_ids,
                    normal_axis=int(ctx.classification.normal_axis),
                )
                cmp = compare_graph_vs_delaunay(graph_sites, d_sites)
                ov3 = cmp["3c"]["overlap"]
                ov4 = cmp["4c"]["overlap"]
            except Exception:
                ov3 = -1
                ov4 = -1
            dmin = min_surface_center_distance(primitives, slab, ctx.detection.surface_atom_ids)
            zmin, zmax, zspan, zlevels = z_level_stats(slab, ctx.detection.surface_atom_ids, int(ctx.classification.normal_axis))
            target_n = ctx.detection.diagnostics.get("target_surface_count", np.nan)
            err = (
                len(ctx.detection.surface_atom_ids) - int(target_n)
                if isinstance(target_n, (int, np.integer))
                else np.nan
            )
            rows.append(
                {
                    "case": case_name,
                    "family": case_family(case_name),
                    "grid_step": cfg["grid_step"],
                    "spacing": cfg["spacing"],
                    "surface_n": len(ctx.detection.surface_atom_ids),
                    "detector": ctx.detection.diagnostics.get("method", ""),
                    "selected_detector": ctx.detection.diagnostics.get("selected_detector", ""),
                    "target_mode": ctx.detection.diagnostics.get("target_mode", ""),
                    "target_rule": ctx.detection.diagnostics.get("target_rule", ""),
                    "target_fraction": ctx.detection.diagnostics.get("target_fraction", np.nan),
                    "estimated_layers": ctx.detection.diagnostics.get("estimated_layers", np.nan),
                    "p1": kinds.get("1c", 0),
                    "p2": kinds.get("2c", 0),
                    "p3": kinds.get("3c", 0),
                    "p4": kinds.get("4c", 0),
                    "dmin_hollow_to_surface": dmin,
                    "delaunay_overlap_3c": ov3,
                    "delaunay_overlap_4c": ov4,
                    "z_min": zmin,
                    "z_max": zmax,
                    "z_span": zspan,
                    "z_level_count": zlevels,
                    "target_surface_count": target_n,
                    "surface_count_error": err,
                    "basis_n": site_dict["meta"]["n_basis_groups"],
                    "basis_compression_ratio": emb_res.compression_ratio,
                    "feature_backend": feature_backend,
                }
            )
            case_dir = out_dir / case_name / f"g{cfg['grid_step']}_s{cfg['spacing']}"
            files = export_surface_detection_report(slab, ctx, case_dir)
            _ = files
            png_path = plot_surface_primitives_2d(slab, ctx, primitives, case_dir / "sites.png")
            png_site_only = plot_site_centers_only(slab, primitives, case_dir / "sites_only.png")
            png_ineq = plot_inequivalent_sites_2d(slab, emb_res.primitives, case_dir / "sites_inequivalent.png")
            png_delta = abs(png_path.stat().st_mtime - png_site_only.stat().st_mtime)
            audit_csv = case_dir / "site_audit.csv"
            with audit_csv.open("w", encoding="utf-8") as f:
                f.write("kind,atom_ids,cx,cy,cz,dmin_surface_mic\n")
                for p in primitives:
                    dmin_site = builder._point_to_surface_min_distance_mic(slab, p.center, ctx.detection.surface_atom_ids)
                    atom_ids = "-".join(str(i) for i in p.atom_ids)
                    f.write(f"{p.kind},{atom_ids},{p.center[0]:.8f},{p.center[1]:.8f},{p.center[2]:.8f},{dmin_site:.8f}\n")
            report_lines.append(f"## {case_name} | grid={cfg['grid_step']} spacing={cfg['spacing']}")
            report_lines.append(f"- surface_n: {len(ctx.detection.surface_atom_ids)}")
            report_lines.append(f"- target_surface_n: {target_n}")
            report_lines.append(f"- detector: {ctx.detection.diagnostics.get('method','')}")
            report_lines.append(f"- selected_detector: {ctx.detection.diagnostics.get('selected_detector','')}")
            report_lines.append(f"- target_mode: {ctx.detection.diagnostics.get('target_mode','')}")
            report_lines.append(f"- target_rule: {ctx.detection.diagnostics.get('target_rule','')}")
            report_lines.append(f"- target_fraction: {ctx.detection.diagnostics.get('target_fraction',np.nan)}")
            report_lines.append(f"- estimated_layers: {ctx.detection.diagnostics.get('estimated_layers',np.nan)}")
            report_lines.append(f"- primitives: 1c={kinds.get('1c',0)} 2c={kinds.get('2c',0)} 3c={kinds.get('3c',0)} 4c={kinds.get('4c',0)}")
            report_lines.append(f"- inequivalent_basis_n: {site_dict['meta']['n_basis_groups']}")
            report_lines.append(f"- basis_compression_ratio: {emb_res.compression_ratio}")
            report_lines.append(f"- feature_backend: {feature_backend}")
            report_lines.append(f"- dmin_hollow_to_surface: {dmin}")
            report_lines.append(f"- z_levels: count={zlevels} min={zmin} max={zmax} span={zspan}")
            report_lines.append(f"- png_mtime_delta_sec: {png_delta:.6f}")
            report_lines.append(f"- png: {png_path.as_posix()}")
            report_lines.append(f"- png_site_only: {png_site_only.as_posix()}")
            report_lines.append(f"- png_inequivalent: {png_ineq.as_posix()}")
            report_lines.append(f"- audit: {audit_csv.as_posix()}")
            report_lines.append("")
    with csv.open("w", encoding="utf-8") as f:
        headers = [
            "case",
            "family",
            "grid_step",
            "spacing",
            "surface_n",
            "detector",
            "selected_detector",
            "target_mode",
            "target_rule",
            "target_fraction",
            "estimated_layers",
            "p1",
            "p2",
            "p3",
            "p4",
            "dmin_hollow_to_surface",
            "delaunay_overlap_3c",
            "delaunay_overlap_4c",
            "z_min",
            "z_max",
            "z_span",
            "z_level_count",
            "target_surface_count",
            "surface_count_error",
            "basis_n",
            "basis_compression_ratio",
            "feature_backend",
        ]
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    with (out_dir / "surface_sweep_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "latest_run.txt").open("w", encoding="utf-8") as f:
        f.write(out_dir.as_posix() + "\n")
    print(csv.resolve())


if __name__ == "__main__":
    main()
