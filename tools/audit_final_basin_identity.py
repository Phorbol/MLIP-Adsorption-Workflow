from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _binding_adsorbate_indices(basin: dict[str, Any]) -> list[int]:
    explicit = basin.get("binding_adsorbate_indices", [])
    if explicit:
        return sorted({int(x) for x in explicit})
    pairs = basin.get("binding_pairs", [])
    return sorted({int(i) for i, _ in pairs})


def analyze_validation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    case_audits = [analyze_case_row(row) for row in rows]
    extra_cases = [x for x in case_audits if int(x["n_extra_ours_basins"]) > 0]
    missed_cases = [x for x in case_audits if int(x["n_missing_manual_basins"]) > 0]
    anchor_extra = sum(
        1
        for case in case_audits
        for basin in case["extra_ours_basins"]
        if str(basin["classification"]["primary_reason"]) == "binding_atom_outside_manual_anchor_set"
    )
    return {
        "source_out_root": payload.get("out_root", ""),
        "n_cases": int(len(case_audits)),
        "n_extra_cases": int(len(extra_cases)),
        "n_missed_cases": int(len(missed_cases)),
        "n_anchor_free_extra_binding_atom_basins": int(anchor_extra),
        "case_audits": case_audits,
    }


def analyze_case_row(row: dict[str, Any]) -> dict[str, Any]:
    ours_work = Path(str(row["ours"]["work_dir"]))
    manual_work = Path(str(row["ase_manual"]["work_dir"]))
    ours = json.loads((ours_work / "basins.json").read_text(encoding="utf-8"))
    manual = json.loads((manual_work / "manual_basin_summary.json").read_text(encoding="utf-8"))

    overlap = dict(row.get("overlap", {}))
    matched_ours_indices = {int(m["ours_index"]) for m in overlap.get("matches", [])}
    manual_anchor_indices = sorted(
        {
            int(site.get("mol_index"))
            for site in manual.get("manual_sites", [])
            if site.get("mol_index", None) is not None
        }
    )
    manual_site_names = sorted(
        {
            str(site.get("site_name"))
            for site in manual.get("manual_sites", [])
            if str(site.get("site_name", "")).strip()
        }
    )

    extra_ours_basins = []
    for basin_idx, basin in enumerate(ours.get("basins", [])):
        if int(basin_idx) in matched_ours_indices:
            continue
        binding_idx = _binding_adsorbate_indices(basin)
        extra_ours_basins.append(
            {
                "basin_index": int(basin_idx),
                "basin_id": int(basin.get("basin_id", basin_idx)),
                "energy_ev": float(basin.get("energy_ev")),
                "signature": str(basin.get("signature", "")),
                "binding_adsorbate_indices": [int(x) for x in binding_idx],
                "binding_adsorbate_symbols": [str(x) for x in basin.get("binding_adsorbate_symbols", [])],
                "binding_pairs": [[int(i), int(j)] for i, j in basin.get("binding_pairs", [])],
                "member_site_labels": [str(x) for x in basin.get("member_site_labels", [])],
                "classification": classify_extra_basin(
                    basin=basin,
                    manual_anchor_indices=manual_anchor_indices,
                    manual_site_names=manual_site_names,
                ),
            }
        )

    n_manual = int(overlap.get("n_manual_basins", row["ase_manual"]["n_basins"]))
    matched_manual = int(overlap.get("matched_manual_basins", 0))
    return {
        "case": str(row["case"]),
        "manual_anchor_indices": [int(x) for x in manual_anchor_indices],
        "manual_site_names": [str(x) for x in manual_site_names],
        "n_manual_basins": int(n_manual),
        "n_ours_basins": int(row["ours"]["n_basins"]),
        "n_matched_manual_basins": int(matched_manual),
        "n_missing_manual_basins": int(max(0, n_manual - matched_manual)),
        "n_extra_ours_basins": int(len(extra_ours_basins)),
        "manual_recall_by_ours": (
            None
            if overlap.get("manual_recall_by_ours", None) is None
            else float(overlap.get("manual_recall_by_ours"))
        ),
        "extra_ours_basins": extra_ours_basins,
    }


def classify_extra_basin(
    *,
    basin: dict[str, Any],
    manual_anchor_indices: list[int],
    manual_site_names: list[str],
) -> dict[str, Any]:
    binding_idx = _binding_adsorbate_indices(basin)
    site_labels = [str(x) for x in basin.get("member_site_labels", [])]
    outside_anchor = bool(binding_idx) and any(int(i) not in set(manual_anchor_indices) for i in binding_idx)
    unseen_site = bool(site_labels) and any(str(x) not in set(manual_site_names) for x in site_labels)
    if outside_anchor:
        reason = "binding_atom_outside_manual_anchor_set"
    elif unseen_site:
        reason = "site_label_outside_manual_reference"
    else:
        reason = "same_anchor_same_site_unmatched"
    return {
        "primary_reason": str(reason),
        "outside_manual_anchor_set": bool(outside_anchor),
        "outside_manual_site_set": bool(unseen_site),
    }


def write_audit_report(audit: dict[str, Any], out_root: Path) -> tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "final_basin_identity_audit.json"
    md_path = out_root / "final_basin_identity_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Final Basin Identity Audit",
        "",
        f"- Cases: {audit['n_cases']}",
        f"- Cases with extra ours basins: {audit['n_extra_cases']}",
        f"- Cases with missed manual basins: {audit['n_missed_cases']}",
        f"- Extra basins caused by non-manual anchor atoms: {audit['n_anchor_free_extra_binding_atom_basins']}",
        "",
        "## Cases",
        "",
    ]
    for case in audit["case_audits"]:
        recall_text = "NA" if case["manual_recall_by_ours"] is None else f"{float(case['manual_recall_by_ours']):.3f}"
        lines.append(
            f"- {case['case']}: manual {case['n_manual_basins']}, ours {case['n_ours_basins']}, "
            f"missing {case['n_missing_manual_basins']}, extra {case['n_extra_ours_basins']}, "
            f"recall {recall_text}"
        )
        for basin in case["extra_ours_basins"]:
            lines.append(
                f"  extra basin {basin['basin_id']}: reason={basin['classification']['primary_reason']}, "
                f"binding_adsorbate_indices={basin['binding_adsorbate_indices']}, "
                f"binding_adsorbate_symbols={basin['binding_adsorbate_symbols']}, "
                f"member_site_labels={basin['member_site_labels']}, energy_ev={basin['energy_ev']}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-json", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.validation_json).read_text(encoding="utf-8"))
    audit = analyze_validation_payload(payload)
    json_path, md_path = write_audit_report(audit, Path(args.out_root))
    print(json_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
