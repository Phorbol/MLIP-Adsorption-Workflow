from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.build import fcc100, fcc111
from ase.collections import g2

from adsorption_ensemble.benchmark import summarize_pose_frames
from adsorption_ensemble.relax.backends import IdentityRelaxBackend, MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import make_adsorption_workflow_config, run_adsorption_workflow
from tests.chemistry_cases import get_test_adsorbate_cases
from tools.run_ase_autoadsorbate_crosscheck import build_slab_suite


def build_audit_slab_suite() -> dict[str, Atoms]:
    slabs = dict(build_slab_suite())
    slabs.setdefault("Cu_fcc111", fcc111("Cu", size=(4, 4, 4), vacuum=12.0))
    slabs.setdefault("Cu_fcc100", fcc100("Cu", size=(4, 4, 4), vacuum=12.0))
    return slabs


def build_audit_adsorbate_suite(max_atoms: int = 40) -> dict[str, Atoms]:
    suite: dict[str, Atoms] = {
        "H": Atoms("H", positions=[[0.0, 0.0, 0.0]]),
    }
    for name in sorted(g2.names):
        try:
            ads = g2[name].copy()
        except Exception:
            continue
        if len(ads) <= int(max_atoms):
            suite.setdefault(str(name), ads)
    for name, ads in get_test_adsorbate_cases().items():
        if len(ads) <= int(max_atoms):
            suite[str(name)] = ads.copy()
    return suite


def parse_case_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for token in tokens:
        if ":" not in str(token):
            raise ValueError(f"Case token must be slab:adsorbate, got {token!r}")
        slab_name, ads_name = str(token).split(":", 1)
        out.append((str(slab_name), str(ads_name)))
    return out


def make_relax_backend(kind: str) -> object:
    if str(kind).strip().lower() == "fake":
        return IdentityRelaxBackend()
    return MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path="/root/.cache/mace/mace-omat-0-small.model",
            device="cuda",
            dtype="float32",
            max_edges_per_batch=20000,
            head_name="omat_pbe",
            enable_cueq=True,
            strict=True,
        )
    )


def run_case(
    *,
    slab_name: str,
    ads_name: str,
    slab: Atoms,
    adsorbate: Atoms,
    out_dir: Path,
    exhaustive: bool,
    placement_mode: str,
    relax_backend: object,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_adsorption_workflow_config(
        out_dir,
        placement_mode=str(placement_mode),
        single_atom=(len(adsorbate) == 1),
        exhaustive_pose_sampling=bool(exhaustive),
        basin_overrides={
            "mace_model_path": "/root/.cache/mace/mace-omat-0-small.model",
            "mace_device": "cuda",
            "mace_dtype": "float32",
            "mace_head_name": "omat_pbe",
            "final_basin_merge_metric": "mace_node_l2",
            "final_basin_merge_node_l2_threshold": 0.20,
            "relax_steps": 80,
            "energy_window_ev": 2.5,
        },
    )
    result = run_adsorption_workflow(
        slab=slab,
        adsorbate=adsorbate,
        config=cfg,
        basin_relax_backend=relax_backend,
    )
    audit = {
        "case": {"slab": str(slab_name), "adsorbate": str(ads_name)},
        "workflow_summary": dict(result.summary),
        "pose_summary": summarize_pose_frames(
            result.pose_frames,
            slab_n=len(slab),
            primitives=result.primitives,
        ),
        "artifacts": dict(result.artifacts),
        "n_basins": int(len(result.basin_result.basins)),
        "n_nodes": int(len(result.nodes)),
        "relax_backend": str(result.basin_result.relax_backend),
    }
    audit_path = out_dir / "pose_sampler_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run workflow-backed pose sampler audits and save rich artifacts.")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["Cu_fcc111:CO", "Cu_fcc111:H2O", "Pt_fcc111:NH3"],
        help="Case tokens in slab:adsorbate form.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/autoresearch/pose_sampler_audit"))
    parser.add_argument("--placement-mode", type=str, default="anchor_free", choices=["anchor_free", "anchor_aware"])
    parser.add_argument("--relax-backend", type=str, default="mace", choices=["mace", "fake"])
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--max-atoms", type=int, default=40)
    args = parser.parse_args()

    slabs = build_audit_slab_suite()
    adsorbates = build_audit_adsorbate_suite(max_atoms=int(args.max_atoms))
    relax_backend = make_relax_backend(str(args.relax_backend))
    rows = []
    for slab_name, ads_name in parse_case_tokens(list(args.cases)):
        if slab_name not in slabs:
            raise ValueError(f"Unknown slab case: {slab_name}")
        if ads_name not in adsorbates:
            raise ValueError(f"Unknown adsorbate case: {ads_name}")
        case_dir = Path(args.out_root) / slab_name / ads_name
        audit = run_case(
            slab_name=slab_name,
            ads_name=ads_name,
            slab=slabs[slab_name].copy(),
            adsorbate=adsorbates[ads_name].copy(),
            out_dir=case_dir,
            exhaustive=bool(args.exhaustive),
            placement_mode=str(args.placement_mode),
            relax_backend=relax_backend,
        )
        rows.append(
            {
                "slab": str(slab_name),
                "adsorbate": str(ads_name),
                "n_pose_frames": int(audit["pose_summary"]["n_pose_frames"]),
                "orientation_bin_counts": dict(audit["pose_summary"]["orientation_bin_counts"]),
                "n_basins": int(audit["n_basins"]),
                "n_nodes": int(audit["n_nodes"]),
                "relax_backend": str(audit["relax_backend"]),
                "audit_json": (case_dir / "pose_sampler_audit.json").as_posix(),
            }
        )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "pose_sampler_audit_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
