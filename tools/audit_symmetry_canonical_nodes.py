from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from adsorption_ensemble.basin.dedup import merge_basin_representatives_by_mace_node_l2
from adsorption_ensemble.basin.reporting import build_node_inflation_audit
from adsorption_ensemble.basin.types import Basin
from adsorption_ensemble.node import NodeConfig


def _load_case(case_dir: Path) -> tuple[dict[str, Any], list[Atoms], list[dict[str, Any]], Atoms, list[Basin]]:
    basin_dict = json.loads((case_dir / "basin_dictionary.json").read_text(encoding="utf-8"))
    nodes = json.loads((case_dir / "nodes.json").read_text(encoding="utf-8"))
    basins = read(case_dir / "basins.extxyz", index=":")
    slab = read(case_dir / "slab_input.xyz")
    if len(basins) != len(basin_dict.get("basins", [])):
        raise ValueError(f"Frame count mismatch in {case_dir}: basins.extxyz={len(basins)} json={len(basin_dict.get('basins', []))}")
    if len(nodes) != len(basin_dict.get("basins", [])):
        raise ValueError(f"Node count mismatch in {case_dir}: nodes.json={len(nodes)} json={len(basin_dict.get('basins', []))}")
    basin_objs = []
    for basin_row, frame in zip(basin_dict.get("basins", []), basins, strict=True):
        basin_objs.append(
            Basin(
                basin_id=int(basin_row["basin_id"]),
                atoms=frame,
                energy_ev=float(basin_row["energy_ev"]),
                member_candidate_ids=[int(x) for x in basin_row.get("member_candidate_ids", [])],
                binding_pairs=[(int(i), int(j)) for i, j in basin_row.get("binding_pairs", [])],
                denticity=int(basin_row.get("denticity", 0)),
                signature=str(basin_row.get("signature", "")),
            )
        )
    return basin_dict, basins, nodes, slab, basin_objs


def _group_rows(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda kv: (min(float(x["energy_ev"]) for x in kv[1]), kv[0])):
        energies = [float(x["energy_ev"]) for x in group]
        out.append(
            {
                "key": str(key),
                "count": int(len(group)),
                "basin_ids": [int(x["basin_id"]) for x in group],
                "node_ids": [str(x["node_id"]) for x in group],
                "signatures": sorted({str(x["signature"]) for x in group}),
                "energy_min_ev": float(min(energies)),
                "energy_max_ev": float(max(energies)),
                "energy_span_ev": float(max(energies) - min(energies)),
                "relative_energies_ev": [float(x["relative_energy_ev"]) for x in group],
            }
        )
    return out


def _threshold_merge_groups(
    *,
    basin_rows: list[dict[str, Any]],
    basin_frames: list[Atoms],
    slab_n: int,
    surface_reference: Atoms,
    thresholds: list[float],
    model_path: str,
    device: str,
    dtype: str,
    max_edges_per_batch: int,
    layers_to_keep: int,
    head_name: str | None,
    mlp_energy_key: str | None,
    enable_cueq: bool,
    use_signature_grouping: bool,
) -> dict[str, Any]:
    basins_payload = []
    for row, frame in zip(basin_rows, basin_frames, strict=True):
        basins_payload.append(
            {
                "basin_id": int(row["basin_id"]),
                "atoms": frame,
                "energy": float(row["energy_ev"]),
                "member_candidate_ids": [int(x) for x in row.get("member_candidate_ids", [])],
                "binding_pairs": [(int(i), int(j)) for i, j in row.get("binding_pairs", [])],
                "signature": str(row["signature"]),
            }
        )

    from adsorption_ensemble.conformer_md.config import MACEInferenceConfig
    from adsorption_ensemble.conformer_md.mace_inference import MACEBatchInferencer

    cfg = MACEInferenceConfig(
        model_path=str(model_path),
        device=str(device),
        dtype=str(dtype),
        enable_cueq=bool(enable_cueq),
        max_edges_per_batch=int(max_edges_per_batch),
        num_workers=1,
        layers_to_keep=int(layers_to_keep),
        mlp_energy_key=(str(mlp_energy_key) if mlp_energy_key else None),
        head_name=str(head_name) if head_name is not None and str(head_name).strip() else "Default",
    )
    infer = MACEBatchInferencer(cfg)
    node_descriptors, _, descriptor_meta = infer.infer_node_descriptors(basin_frames)

    out: dict[str, Any] = {
        "descriptor_meta": dict(descriptor_meta),
        "thresholds": {},
    }
    for thr in thresholds:
        merged, meta = merge_basin_representatives_by_mace_node_l2(
            basins=basins_payload,
            slab_n=int(slab_n),
            binding_tau=1.20,
            node_l2_threshold=float(thr),
            mace_model_path=str(model_path),
            mace_device=str(device),
            mace_dtype=str(dtype),
            mace_enable_cueq=bool(enable_cueq),
            mace_max_edges_per_batch=int(max_edges_per_batch),
            mace_layers_to_keep=int(layers_to_keep),
            mace_head_name=head_name,
            mace_mlp_energy_key=mlp_energy_key,
            cluster_method="hierarchical",
            l2_mode="mean_atom",
            node_descriptors=node_descriptors,
            signature_mode="reference_canonical",
            use_signature_grouping=bool(use_signature_grouping),
            surface_reference=surface_reference.copy(),
            energy_gate_ev=None,
        )
        groups = []
        for merged_row in merged:
            src = [int(x) for x in merged_row.get("source_basin_ids", [])]
            source_rows = [basin_rows[i] for i in src]
            energies = [float(x["energy_ev"]) for x in source_rows]
            observed_node_ids = sorted({str(x["node_id"]) for x in source_rows})
            legacy_node_ids = sorted({str(x["node_id_legacy"]) for x in source_rows})
            surface_geometry_node_ids = sorted({str(x["node_id_surface_geometry"]) for x in source_rows})
            surface_env_only_keys = sorted({str(x["surface_env_only_key"]) for x in source_rows})
            surface_geometry_keys = sorted({str(x["surface_geometry_key"]) for x in source_rows})
            groups.append(
                {
                    "merged_basin_id": int(merged_row["basin_id"]),
                    "source_basin_ids": src,
                    "source_node_ids": [str(x["node_id"]) for x in source_rows],
                    "source_observed_node_ids": observed_node_ids,
                    "source_legacy_node_ids": legacy_node_ids,
                    "source_surface_geometry_node_ids": surface_geometry_node_ids,
                    "source_signatures": sorted({str(x["signature"]) for x in source_rows}),
                    "surface_env_only_keys": surface_env_only_keys,
                    "surface_geometry_keys": surface_geometry_keys,
                    "n_unique_observed_node_ids": int(len(observed_node_ids)),
                    "n_unique_legacy_node_ids": int(len(legacy_node_ids)),
                    "n_unique_surface_geometry_node_ids": int(len(surface_geometry_node_ids)),
                    "n_unique_surface_env_only_keys": int(len(surface_env_only_keys)),
                    "n_unique_surface_geometry_keys": int(len(surface_geometry_keys)),
                    "energy_min_ev": float(min(energies)),
                    "energy_max_ev": float(max(energies)),
                    "energy_span_ev": float(max(energies) - min(energies)),
                    "relative_energies_ev": [float(x["relative_energy_ev"]) for x in source_rows],
                }
            )
        out["thresholds"][f"{float(thr):.2f}"] = {
            "n_output_basins": int(len(merged)),
            "use_signature_grouping": bool(use_signature_grouping),
            "meta": dict(meta),
            "groups": groups,
        }
    return out


def _summarize_threshold_node_audit(
    *,
    threshold_payload: dict[str, Any],
    observed_node_count: int,
    legacy_node_count: int,
    surface_geometry_node_count: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for thr, payload in sorted(threshold_payload.get("thresholds", {}).items(), key=lambda kv: float(kv[0])):
        groups = list(payload.get("groups", []))
        same_geom_resolutions = 0
        cross_geom_merges = 0
        cross_geom_basin_ids: list[int] = []
        for group in groups:
            n_observed = int(group.get("n_unique_observed_node_ids", 0))
            n_geom = int(group.get("n_unique_surface_geometry_keys", 0))
            if n_geom == 1 and n_observed > 1:
                same_geom_resolutions += 1
            if n_geom > 1:
                cross_geom_merges += 1
                cross_geom_basin_ids.extend(int(x) for x in group.get("source_basin_ids", []))
        n_output = int(payload.get("n_output_basins", 0))
        out[str(thr)] = {
            "n_output_basins": n_output,
            "delta_vs_observed_node_ids": int(n_output - observed_node_count),
            "delta_vs_legacy_node_ids": int(n_output - legacy_node_count),
            "delta_vs_surface_geometry_node_ids": int(n_output - surface_geometry_node_count),
            "same_surface_geometry_resolutions": int(same_geom_resolutions),
            "cross_surface_geometry_merges": int(cross_geom_merges),
            "cross_surface_geometry_basin_ids": sorted(set(cross_geom_basin_ids)),
        }
    return out


def analyze_case(
    *,
    case_dir: Path,
    thresholds: list[float],
    device: str,
) -> dict[str, Any]:
    basin_dict, basin_frames, nodes, slab, basin_objs = _load_case(case_dir)
    basins_json = list(basin_dict.get("basins", []))
    summary = dict(basin_dict.get("summary", {}))
    final_merge = dict(summary.get("final_basin_merge", {}))
    slab_n = len(slab)
    energy_min = float(summary.get("energy_min_ev"))
    inflation_audit = build_node_inflation_audit(
        basins=basin_objs,
        slab_n=slab_n,
        surface_reference=slab,
        observed_nodes=nodes,
        node_cfg=NodeConfig(node_identity_mode="surface_geometry"),
    )
    rows = []
    legacy_rows = {int(r["basin_id"]): r for r in inflation_audit["rows_by_mode"].get("legacy_absolute", [])}
    geom_rows = {int(r["basin_id"]): r for r in inflation_audit["rows_by_mode"].get("surface_geometry", [])}
    observed_rows = {int(r["basin_id"]): r for r in inflation_audit["rows_by_mode"].get("observed", [])}
    for basin_row in basins_json:
        basin_id = int(basin_row["basin_id"])
        obs_row = observed_rows.get(basin_id, {})
        legacy_row = legacy_rows.get(basin_id, {})
        geom_row = geom_rows.get(basin_id, {})
        row = {
            "basin_id": basin_id,
            "node_id": str(obs_row.get("node_id", "")),
            "node_id_legacy": str(legacy_row.get("node_id", obs_row.get("node_id_legacy", ""))),
            "node_id_surface_geometry": str(geom_row.get("node_id", "")),
            "signature": str(basin_row["signature"]),
            "energy_ev": float(basin_row["energy_ev"]),
            "relative_energy_ev": float(basin_row["energy_ev"]) - energy_min,
            "binding_pairs": [[int(i), int(j)] for i, j in basin_row.get("binding_pairs", [])],
            "member_candidate_ids": [int(x) for x in basin_row.get("member_candidate_ids", [])],
            "surface_env_only_key": str(geom_row.get("surface_env_key", "")),
            "surface_geometry_key": str(geom_row.get("surface_geometry_key", "")),
        }
        rows.append(row)

    model_path = str(final_merge.get("model_path", ""))
    if not model_path:
        raise ValueError(f"No final merge model path found in {case_dir}")

    threshold_grouped = _threshold_merge_groups(
        basin_rows=rows,
        basin_frames=basin_frames,
        slab_n=slab_n,
        surface_reference=slab,
        thresholds=thresholds,
        model_path=model_path,
        device=str(device),
        dtype=str(final_merge.get("dtype", "float32")),
        max_edges_per_batch=int(final_merge.get("max_edges_per_batch", 15000)),
        layers_to_keep=int(final_merge.get("layers_to_keep", -1)),
        head_name=final_merge.get("head_name"),
        mlp_energy_key=final_merge.get("mlp_energy_key"),
        enable_cueq=bool(final_merge.get("enable_cueq", False)),
        use_signature_grouping=True,
    )
    threshold_ungrouped = _threshold_merge_groups(
        basin_rows=rows,
        basin_frames=basin_frames,
        slab_n=slab_n,
        surface_reference=slab,
        thresholds=thresholds,
        model_path=model_path,
        device=str(device),
        dtype=str(final_merge.get("dtype", "float32")),
        max_edges_per_batch=int(final_merge.get("max_edges_per_batch", 15000)),
        layers_to_keep=int(final_merge.get("layers_to_keep", -1)),
        head_name=final_merge.get("head_name"),
        mlp_energy_key=final_merge.get("mlp_energy_key"),
        enable_cueq=bool(final_merge.get("enable_cueq", False)),
        use_signature_grouping=False,
    )

    env_only_groups = _group_rows(rows, "surface_env_only_key")
    geom_groups = _group_rows(rows, "surface_geometry_key")
    signature_groups = _group_rows(rows, "signature")
    current_node_groups = _group_rows(rows, "node_id")
    legacy_node_groups = _group_rows(rows, "node_id_legacy")
    surface_mode_groups = _group_rows(rows, "node_id_surface_geometry")
    downstream_node_inflation_audit = {
        "grouped": _summarize_threshold_node_audit(
            threshold_payload=threshold_grouped,
            observed_node_count=int(len(current_node_groups)),
            legacy_node_count=int(len(legacy_node_groups)),
            surface_geometry_node_count=int(len(surface_mode_groups)),
        ),
        "ungrouped": _summarize_threshold_node_audit(
            threshold_payload=threshold_ungrouped,
            observed_node_count=int(len(current_node_groups)),
            legacy_node_count=int(len(legacy_node_groups)),
            surface_geometry_node_count=int(len(surface_mode_groups)),
        ),
    }
    return {
        "case": str(case_dir.name),
        "case_dir": case_dir.as_posix(),
        "slab_n": int(slab_n),
        "n_basins_original": int(len(rows)),
        "false_split_suspect_signatures": [str(x) for x in basin_dict.get("false_split_suspect_signatures", [])],
        "counts": {
            "current_node_ids": int(len(current_node_groups)),
            "legacy_node_ids": int(len(legacy_node_groups)),
            "surface_geometry_node_ids": int(len(surface_mode_groups)),
            "inflation_observed_vs_surface_geometry": int(
                inflation_audit["counts"].get("inflation_observed_vs_surface_geometry", 0)
            ),
            "inflation_legacy_vs_surface_geometry": int(
                inflation_audit["counts"].get("inflation_legacy_vs_surface_geometry", 0)
            ),
            "redundant_surface_geometry_split_groups": int(
                inflation_audit["counts"].get("redundant_surface_geometry_split_groups", 0)
            ),
            "redundant_surface_geometry_split_basins": int(
                inflation_audit["counts"].get("redundant_surface_geometry_split_basins", 0)
            ),
            "signatures": int(len(signature_groups)),
            "surface_env_only_keys": int(len(env_only_groups)),
            "surface_geometry_keys": int(len(geom_groups)),
            "threshold_0p05_grouped": int(threshold_grouped["thresholds"].get("0.05", {}).get("n_output_basins", -1)),
            "threshold_0p08_grouped": int(threshold_grouped["thresholds"].get("0.08", {}).get("n_output_basins", -1)),
            "threshold_0p05_ungrouped": int(threshold_ungrouped["thresholds"].get("0.05", {}).get("n_output_basins", -1)),
            "threshold_0p08_ungrouped": int(threshold_ungrouped["thresholds"].get("0.08", {}).get("n_output_basins", -1)),
        },
        "node_inflation_audit": inflation_audit,
        "current_nodes": current_node_groups,
        "legacy_nodes": legacy_node_groups,
        "surface_geometry_nodes": surface_mode_groups,
        "signature_groups": signature_groups,
        "surface_env_only_groups": env_only_groups,
        "surface_geometry_groups": geom_groups,
        "threshold_merge_audit": {
            "grouped": threshold_grouped,
            "ungrouped": threshold_ungrouped,
        },
        "downstream_node_inflation_audit": downstream_node_inflation_audit,
        "basins": rows,
    }


def analyze_run(*, run_dir: Path, thresholds: list[float], device: str) -> dict[str, Any]:
    outputs_root = run_dir / "outputs"
    cases = []
    for case_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir() and (p / "basin_dictionary.json").exists()):
        cases.append(analyze_case(case_dir=case_dir, thresholds=thresholds, device=device))
    return {
        "run_dir": run_dir.as_posix(),
        "device": str(device),
        "thresholds": [float(x) for x in thresholds],
        "cases": cases,
    }


def write_report(report: dict[str, Any], out_root: Path) -> tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "symmetry_canonical_node_audit.json"
    md_path = out_root / "symmetry_canonical_node_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Symmetry-Canonical Node Audit",
        "",
        f"- Run: `{report['run_dir']}`",
        f"- Device: `{report['device']}`",
        f"- Thresholds: {', '.join(f'{float(x):.2f}' for x in report['thresholds'])}",
        "",
    ]
    for case in report["cases"]:
        counts = case["counts"]
        lines.extend(
            [
                f"## {case['case']}",
                "",
                f"- Original basins / observed nodes: `{case['n_basins_original']}` / `{counts['current_node_ids']}`",
                f"- Recomputed node counts: legacy `{counts['legacy_node_ids']}`, surface-geometry `{counts['surface_geometry_node_ids']}`",
                f"- Node inflation vs surface-geometry: observed `{counts['inflation_observed_vs_surface_geometry']}`, legacy `{counts['inflation_legacy_vs_surface_geometry']}`",
                f"- Signature groups: `{counts['signatures']}`",
                f"- Symmetry node counts: env-only `{counts['surface_env_only_keys']}`, surface-geometry `{counts['surface_geometry_keys']}`",
                f"- Final-merge audited counts (grouped): `0.05 -> {counts['threshold_0p05_grouped']}`, `0.08 -> {counts['threshold_0p08_grouped']}`",
                f"- Final-merge audited counts (ungrouped): `0.05 -> {counts['threshold_0p05_ungrouped']}`, `0.08 -> {counts['threshold_0p08_ungrouped']}`",
                f"- False-split suspect signatures: {', '.join(case['false_split_suspect_signatures']) if case['false_split_suspect_signatures'] else 'none'}",
                "",
                "### Node Inflation Audit",
                "",
            ]
        )
        redundant = case["node_inflation_audit"]["redundant_surface_geometry_splits"]
        lines.append(
            f"- Redundant split groups: `{counts['redundant_surface_geometry_split_groups']}` "
            f"(basins `{counts['redundant_surface_geometry_split_basins']}`)"
        )
        if not redundant:
            lines.append("- Redundant surface-geometry splits: none")
        else:
            for group in redundant:
                lines.append(
                    f"- Redundant surface-geometry split key={group['surface_geometry_key']}: basins={group['basin_ids']}, "
                    f"observed_node_ids={group['observed_node_ids']}, span={group['energy_span_ev']:.6f} eV"
                )
        lines.extend(
            [
                "",
                "### Surface-Geometry Groups",
                "",
            ]
        )
        for group in case["surface_geometry_groups"]:
            lines.append(
                f"- key={group['key']}: basins={group['basin_ids']}, node_ids={group['node_ids']}, "
                f"signatures={group['signatures']}, span={group['energy_span_ev']:.6f} eV"
            )
        lines.extend(["", "### Downstream Dedup Audit", ""])
        for mode_key in ["grouped", "ungrouped"]:
            lines.append(f"- mode `{mode_key}`")
            for thr, payload in sorted(case["downstream_node_inflation_audit"][mode_key].items(), key=lambda kv: float(kv[0])):
                lines.append(
                    f"  threshold `{thr}` -> delta_vs_surface_geometry=`{payload['delta_vs_surface_geometry_node_ids']}`, "
                    f"same_geom_resolutions=`{payload['same_surface_geometry_resolutions']}`, "
                    f"cross_geom_merges=`{payload['cross_surface_geometry_merges']}`"
                )
            lines.append("")
        lines.extend(["### Threshold Merges", ""])
        for mode_key in ["grouped", "ungrouped"]:
            lines.append(f"- mode `{mode_key}`")
            for thr, payload in sorted(case["threshold_merge_audit"][mode_key]["thresholds"].items(), key=lambda kv: float(kv[0])):
                lines.append(f"  threshold `{thr}` -> `{payload['n_output_basins']}` basins")
                for group in payload["groups"]:
                    lines.append(
                        f"    merge {group['merged_basin_id']}: src={group['source_basin_ids']}, "
                        f"observed_node_ids={group['source_observed_node_ids']}, "
                        f"surface_geometry_node_ids={group['source_surface_geometry_node_ids']}, "
                        f"geom_keys={group['surface_geometry_keys']}, "
                        f"span={group['energy_span_ev']:.6f} eV"
                    )
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    thresholds = args.threshold or [0.05, 0.08]
    report = analyze_run(
        run_dir=Path(args.run_dir),
        thresholds=[float(x) for x in thresholds],
        device=str(args.device),
    )
    json_path, md_path = write_report(report, Path(args.out_root))
    print(json_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
