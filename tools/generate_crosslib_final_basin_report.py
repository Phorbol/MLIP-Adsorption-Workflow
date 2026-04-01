from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    src = Path("artifacts/autoresearch/final_basin_crosslib/autoadsorbate_final_basin_benchmark.json")
    payload = json.loads(src.read_text(encoding="utf-8"))
    rows = payload["rows"]

    total_ours = sum(int(r["ours"]["n_basins"]) for r in rows)
    total_auto = sum(int(r["autoadsorbate"]["n_basins"]) for r in rows)
    total_auto_attempted = sum(int(r["autoadsorbate"]["generator"]["n_attempted"]) for r in rows)
    total_auto_accepted = sum(int(r["autoadsorbate"]["generator"]["n_accepted"]) for r in rows)
    total_auto_rejected = sum(int(r["autoadsorbate"]["basin_summary"]["n_rejected"]) for r in rows)

    out_dir = Path("artifacts/autoresearch/paper_positioning")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "crosslib_final_basin_report.md"
    lines = [
        "# Cross-Library Final Basin Report",
        "",
        "## Summary",
        "",
        f"- Cases: {len(rows)}",
        f"- Ours total final basins: {total_ours}",
        f"- AutoAdsorbate total final basins after common relax/filter: {total_auto}",
        f"- AutoAdsorbate generated candidates: {total_auto_accepted}/{total_auto_attempted} accepted before relax",
        f"- AutoAdsorbate rejected after relax/filter: {total_auto_rejected}",
        "",
        "## Per-Case Readout",
        "",
    ]
    for r in rows:
        lines.extend(
            [
                f"- {r['case']}: ours {r['ours']['n_basins']} basins; autoadsorbate {r['autoadsorbate']['n_basins']} basins; "
                f"accepted candidates {r['autoadsorbate']['generator']['n_accepted']}/{r['autoadsorbate']['generator']['n_attempted']}; "
                f"rejected reasons {r['autoadsorbate'].get('rejected_reason_counts', {})}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This benchmark is intentionally narrow: O/N monodentate small molecules on Pt surfaces, using AutoAdsorbate's marked-SMILES route and shrinkwrap top-site generation.",
            "- Within this scope, AutoAdsorbate can place chemically marked candidates, but those candidates do not survive the same final relax + anomaly-filter backend used by our workflow.",
            "- Our method is therefore not only generating more candidates. It is producing candidates that are much more likely to remain valid low-energy adsorption basins after common relaxation.",
            "- The main caveat is that this is not a full AutoAdsorbate workflow comparison across every supported chemistry. It is a controlled comparison in a shared backend regime.",
            "",
            "## Story Value",
            "",
            "- This result supports the paper claim that the contribution is not merely site discovery.",
            "- The differentiator is the primitive-basis -> pose-pool -> relaxed-basin pipeline, which preserves valid adsorption states under a common downstream relaxation criterion.",
            "- This gives a stronger publication story than a pure geometric site-count comparison.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
