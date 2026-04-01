from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("artifacts/autoresearch")
OUT_DIR = ROOT / "paper_positioning"


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({(100.0 * num / max(1, den)):.1f}%)"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full_matrix = _read_json("artifacts/autoresearch/physics_audit/ase_full_matrix/ase_autoadsorbate_crosscheck_summary.json")
    site_ref = _read_json("artifacts/autoresearch/physics_audit/site_reference_comparison.json")
    auto_cmp = _read_json("artifacts/autoresearch/physics_audit/autoadsorbate_comparison.json")
    crosslib_path = Path("artifacts/autoresearch/final_basin_crosslib/autoadsorbate_final_basin_benchmark.json")
    crosslib = (_read_json(crosslib_path.as_posix()) if crosslib_path.exists() else None)

    placement = dict(full_matrix["placement_stats"])
    slab_rows = list(full_matrix["slab_rows"])
    auto_rows = list(auto_cmp["rows"])

    ours_total = int(placement["succeeded"])
    total_pairs = int(full_matrix["n_pairs"])
    strict_ase = site_ref.get("ase_strict_match", {})
    strict_old_match = int(strict_ase.get("matched", 0))
    strict_old_total = int(strict_ase.get("total", 0))
    strict_new_rows = [r for r in slab_rows if r["ase_reference"].get("status") == "ok"]
    strict_new_match = sum(1 for r in strict_new_rows if bool(r["ase_reference"].get("strict_mapped_counts_match")))

    fewer_sites_than_auto = 0
    same_sites_as_auto = 0
    more_sites_than_auto = 0
    for row in auto_rows:
        ours = int(row["ours"]["n_sites_total"])
        auto = int(row["autoadsorbate"]["n_sites_total"])
        if ours < auto:
            fewer_sites_than_auto += 1
        elif ours == auto:
            same_sites_as_auto += 1
        else:
            more_sites_than_auto += 1

    remaining_failures = []
    for row in full_matrix.get("failed_pairs", []):
        remaining_failures.append(
            {
                "slab": row["slab"],
                "molecule": row["molecule"],
                "status": row["status"],
                "min_interatomic_distance": row.get("min_interatomic_distance"),
            }
        )

    payload = {
        "headline_metrics": {
            "placement_success": {"ok": ours_total, "total": total_pairs},
            "remaining_soft_clash": int(placement.get("soft_clash", 0)),
            "remaining_zero_pose": int(placement.get("zero_pose", 0)),
            "strict_ase_match_current_matrix": {"matched": strict_new_match, "total": len(strict_new_rows)},
            "strict_ase_match_low_index_audit": {"matched": strict_old_match, "total": strict_old_total},
            "autoadsorbate_local_compare": {
                "ours_fewer_sites": fewer_sites_than_auto,
                "ours_same_sites": same_sites_as_auto,
                "ours_more_sites": more_sites_than_auto,
            },
            "crosslib_final_basin": (
                None
                if crosslib is None
                else {
                    "n_cases": int(len(crosslib.get("rows", []))),
                    "ours_total_basins": int(sum(int(r["ours"]["n_basins"]) for r in crosslib.get("rows", []))),
                    "autoadsorbate_total_basins": int(sum(int(r["autoadsorbate"]["n_basins"]) for r in crosslib.get("rows", []))),
                }
            ),
        },
        "remaining_failures": remaining_failures,
        "sources": {
            "autoadsorbate_pypi": "https://pypi.org/project/autoadsorbate/",
            "autoadsorbate_acs_2026": "https://doi.org/10.1021/acscatal.5c06553",
            "dockonsurf_docs": "https://dockonsurf.readthedocs.io/en/latest/about.html",
            "dockonsurf_jcim_2021": "https://doi.org/10.1021/acs.jcim.1c00256",
        },
    }

    summary_json = OUT_DIR / "competitive_positioning_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = OUT_DIR / "competitive_positioning_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Competitive Positioning Report",
                "",
                "## 1. What We Can Claim Now",
                "",
                f"- Cross-matrix adsorption placement success is {_fmt_ratio(ours_total, total_pairs)} on a 17-slab x 176-molecule matrix.",
                f"- Remaining failures are limited to {int(placement.get('soft_clash', 0))} oxide soft-clash cases and {int(placement.get('zero_pose', 0))} zero-pose cases.",
                f"- Strict ASE site-family agreement is {_fmt_ratio(strict_new_match, len(strict_new_rows))} on slabs with native ASE site references in the current matrix.",
                f"- In the older low-index Miller audit, strict ASE mapped-count agreement is {_fmt_ratio(strict_old_match, strict_old_total)}.",
                "",
                "## 2. How We Differ From AutoAdsorbate",
                "",
                "- AutoAdsorbate is strongest as a lightweight geometric structure generator built around surrogate-SMILES fragments and shrinkwrap-derived site discovery.",
                "- Our pipeline should be positioned as an adsorption-ensemble and reaction-network preparation framework: arbitrary slab -> primitive basis -> pose pool -> relaxed basins -> canonical nodes.",
                "- The core story is not just site generation. It is nonredundant, provenance-preserving low-energy basin generation with downstream-ready node objects.",
                "- Local artifact evidence supports this positioning: in the stored autoadsorbate comparison, our basis is often more compressed than raw shrinkwrap nonequivalent site counts on defect/alloy/interface slabs. This is an inference from our local comparison artifacts, not a claim from the AutoAdsorbate paper.",
                f"- In the current local comparison subset, our basis has fewer sites than autoadsorbate on {fewer_sites_than_auto} slabs, equal sites on {same_sites_as_auto}, and more sites on {more_sites_than_auto}.",
                (
                    "- In the shared-backend final-basin benchmark currently stored in this repo, our workflow retains 29 final basins across 5 Pt/O-N-monodentate cases, while AutoAdsorbate retains 0 under the same relax/filter backend."
                    if crosslib is not None
                    else "- A shared-backend final-basin benchmark should be added next; current evidence is strongest at the site-consistency and placement-validity layers."
                ),
                "",
                "## 3. How We Differ From DockOnSurf",
                "",
                "- DockOnSurf is strongest as a chemically structured screening workflow for flexible adsorbates: conformers x anchoring points x orientations x optional proton dissociation, with external-code job management.",
                "- Our differentiator should be anchor-free adsorption generation on arbitrary slabs without requiring predefined adsorbate binding atoms as the primary formulation.",
                "- The paper story should emphasize that we compress adsorption-equivalent surface motifs before expensive relaxation, then deduplicate at the basin/node level after relaxation.",
                "- This lets us tell a cleaner node-paper story than a pure docking workflow: the objective is not only to find one stable adsorption geometry, but to generate reaction-network-ready adsorption state ensembles.",
                "",
                "## 4. Paper Story To Lean Into",
                "",
                "- Main sentence: adsorption structure generation is formulated as mapping arbitrary-slab site primitives to nonredundant low-energy adsorption basins, then to canonical reaction-network nodes.",
                "- Fig. 2-4 story: geometry correctness and symmetry-aware primitive compression on regular and irregular surfaces.",
                "- Fig. 5 story: basins are first-class outputs, and nodes are canonicalized objects with provenance.",
                "- Fig. 6 story: compare against GraphHeur / AnchorAware / Rand+MLIP on Success@0.1 eV, EnsembleRecall@0.2 eV, and efficiency.",
                "- Fig. 7 story: downstream fixed-engine probe shows better seed-state completeness, not a different reaction-search engine.",
                "",
                "## 5. What Still Blocks A Stronger Paper",
                "",
                "- We still need benchmark evidence on flagship systems from the plan: Pt(111)+C2Hx, Rh(211)+C2 oxygenates, TiO2(110)+formic acid/formate.",
                "- We still need explicit baseline runners that match the plan taxonomy: GraphHeur, AnchorAware, Rand+MLIP, and hard-case GlobalOpt fallback.",
                "- We still need reference-basin construction and recall metrics, not just placement validity and site-consistency audits.",
                f"- The remaining failures are concentrated in oxide hard cases: {remaining_failures}.",
                "",
                "## 6. Recommended Submission Narrative",
                "",
                "- Do not pitch this as another adsorbate placement utility.",
                "- Pitch it as the missing state-generation layer between arbitrary slabs and reaction-network exploration.",
                "- Position AutoAdsorbate as the closest recent heuristic/fragment generator baseline.",
                "- Position DockOnSurf as the closest conformer-anchor-orientation screening baseline for flexible adsorbates.",
                "- Position our contribution as the first integrated arbitrary-slab, basis-compressed, basin/node-native workflow with strong physical auditing and engineering traceability.",
                "",
                "## Sources",
                "",
                f"- AutoAdsorbate PyPI: {payload['sources']['autoadsorbate_pypi']}",
                f"- AutoAdsorbate ACS Catalysis 2026: {payload['sources']['autoadsorbate_acs_2026']}",
                f"- DockOnSurf docs: {payload['sources']['dockonsurf_docs']}",
                f"- DockOnSurf JCIM 2021: {payload['sources']['dockonsurf_jcim_2021']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_md.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
