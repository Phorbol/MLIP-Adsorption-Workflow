from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig, build_site_dictionary
from adsorption_ensemble.surface import SurfacePreprocessor, export_surface_detection_report
from adsorption_ensemble.visualization import plot_inequivalent_sites_2d, plot_site_centers_only, plot_surface_primitives_2d
from tools.run_ase_autoadsorbate_crosscheck import build_slab_suite


DEFAULT_CASES = [
    "Pt_fcc110",
    "Pt_fcc211",
    "TiO2_110",
    "MgO_100",
    "CuNi_fcc111_alloy",
    "Pt_fcc111_vacancy",
    "Pt_fcc111_adatom",
    "Pt_fcc111_cluster_interface",
]


def _make_summary_record(slab, ctx, emb) -> dict:
    axis = int(ctx.classification.normal_axis)
    coords = np.asarray(slab.get_positions(), dtype=float)
    z = np.round(coords[ctx.detection.surface_atom_ids, axis], 6)
    return {
        "n_atoms": int(len(slab)),
        "n_surface_atoms": int(len(ctx.detection.surface_atom_ids)),
        "surface_z_levels": [float(v) for v in sorted(set(z.tolist()), reverse=True)],
        "surface_symbols": Counter(str(slab[int(i)].symbol) for i in ctx.detection.surface_atom_ids),
        "graph_edge_count": int(len(ctx.graph.edges)),
        "raw_primitive_count": int(emb.raw_count),
        "basis_primitive_count": int(emb.basis_count),
        "basis_kind_counts": {str(k): int(v) for k, v in Counter(p.kind for p in emb.basis_primitives).items()},
        "diagnostics": ctx.detection.diagnostics,
    }


def run_case(case_name: str, slab, out_dir: Path, mode: str) -> dict:
    case_dir = out_dir / case_name / mode
    case_dir.mkdir(parents=True, exist_ok=True)
    pre = SurfacePreprocessor(min_surface_atoms=6, target_count_mode=mode)
    ctx = pre.build_context(slab)
    export_surface_detection_report(slab, ctx, case_dir / "surface_report")

    raw = PrimitiveBuilder().build(slab, ctx)
    atom_features = (slab.get_atomic_numbers().astype(float) / (np.max(slab.get_atomic_numbers()) + 1.0e-12)).reshape(-1, 1)
    emb = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.20)).fit_transform(
        slab=slab,
        primitives=list(raw),
        atom_features=atom_features,
    )

    (case_dir / "raw_site_dictionary.json").write_text(
        json.dumps(build_site_dictionary(emb.primitives, slab=slab), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (case_dir / "selected_site_dictionary.json").write_text(
        json.dumps(build_site_dictionary(emb.basis_primitives, slab=slab), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_surface_primitives_2d(slab=slab, context=ctx, primitives=emb.primitives, filename=case_dir / "sites.png")
    plot_site_centers_only(slab=slab, primitives=emb.primitives, filename=case_dir / "sites_only.png")
    plot_inequivalent_sites_2d(slab=slab, primitives=emb.primitives, filename=case_dir / "sites_inequivalent.png")

    record = _make_summary_record(slab, ctx, emb)
    (case_dir / "summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-audit surface atom detection and site generation on representative slabs.")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/autoresearch/physics_audit/surface_atom_reaudit_20260404"))
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--modes", nargs="*", default=["fixed", "off", "adaptive"])
    args = parser.parse_args()

    slabs = build_slab_suite()
    summary: dict[str, dict[str, dict]] = {}
    for case_name in args.cases:
        if case_name not in slabs:
            raise ValueError(f"Unknown case: {case_name}")
        slab = slabs[case_name]
        summary[case_name] = {}
        for mode in args.modes:
            summary[case_name][mode] = run_case(case_name, slab, args.out_root, str(mode))

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "surface_atom_reaudit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print((args.out_root / "surface_atom_reaudit_summary.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
