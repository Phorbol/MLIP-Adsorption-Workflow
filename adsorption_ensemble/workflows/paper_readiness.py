from __future__ import annotations

from dataclasses import dataclass

from .adsorption import AdsorptionWorkflowResult


@dataclass
class PaperReadinessReport:
    score: int
    max_score: int
    checks: dict[str, bool]
    summary: dict


def evaluate_adsorption_workflow_readiness(result: AdsorptionWorkflowResult) -> PaperReadinessReport:
    checks = {
        "surface_detected": int(result.summary.get("n_surface_atoms", 0)) > 0,
        "primitives_enumerated": int(result.summary.get("n_primitives", 0)) > 0,
        "pose_pool_generated": int(result.summary.get("n_pose_frames", 0)) > 0,
        "basins_generated": int(result.summary.get("n_basins", 0)) > 0,
        "nodes_generated": int(result.summary.get("n_nodes", 0)) > 0,
        "site_dictionary_saved": bool(result.artifacts.get("site_dictionary_json")),
        "basins_json_saved": bool(result.artifacts.get("basins_json")),
        "nodes_json_saved": bool(result.artifacts.get("nodes_json")),
        "flexible_adsorbate_supported": (not bool(result.summary.get("run_conformer_search"))) or int(result.summary.get("n_conformers", 0)) > 0,
    }
    score = int(sum(bool(v) for v in checks.values()))
    return PaperReadinessReport(
        score=score,
        max_score=len(checks),
        checks=checks,
        summary=dict(result.summary),
    )
