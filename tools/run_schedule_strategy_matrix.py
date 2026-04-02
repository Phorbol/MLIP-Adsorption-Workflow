from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import fcc111, fcc100, molecule

from adsorption_ensemble.relax.backends import IdentityRelaxBackend
from adsorption_ensemble.selection import StageSelectionConfig
from adsorption_ensemble.workflows import SamplingSchedule, generate_adsorption_ensemble


def _cases() -> list[tuple[str, Atoms, Atoms]]:
    co = molecule("CO")
    nh3 = molecule("NH3")
    return [
        ("Pt111_CO", fcc111("Pt", size=(3, 3, 4), vacuum=12.0), co),
        ("Pt100_NH3", fcc100("Pt", size=(3, 3, 4), vacuum=12.0), nh3),
    ]


def _pre_strategies() -> dict[str, StageSelectionConfig]:
    return {
        "none": StageSelectionConfig(enabled=False, strategy="none"),
        "fps": StageSelectionConfig(enabled=True, strategy="fps", max_candidates=6, random_seed=0),
        "hierarchical": StageSelectionConfig(enabled=True, strategy="hierarchical", cluster_threshold=0.05),
        "fuzzy": StageSelectionConfig(enabled=True, strategy="fuzzy", cluster_threshold=0.05),
    }


def _post_strategies() -> dict[str, StageSelectionConfig]:
    return {
        "none": StageSelectionConfig(enabled=False, strategy="none"),
        "fps": StageSelectionConfig(enabled=True, strategy="fps", max_candidates=6, random_seed=0),
        "molclus_like": StageSelectionConfig(enabled=True, strategy="energy_rmsd_window", energy_window_ev=1.0, rmsd_threshold=0.05),
        "hierarchical": StageSelectionConfig(enabled=True, strategy="hierarchical", cluster_threshold=0.05),
        "fuzzy": StageSelectionConfig(enabled=True, strategy="fuzzy", cluster_threshold=0.05),
    }


def run_matrix(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pre_strategies = _pre_strategies()
    post_strategies = _post_strategies()
    for case_name, slab, ads in _cases():
        for pre_name, pre_cfg in pre_strategies.items():
            for post_name, post_cfg in post_strategies.items():
                run_dir = out_root / case_name / f"pre_{pre_name}__post_{post_name}"
                result = generate_adsorption_ensemble(
                    slab=slab,
                    adsorbate=ads,
                    work_dir=run_dir,
                    placement_mode="anchor_free",
                    schedule=SamplingSchedule(
                        name="strategy_matrix",
                        pre_relax_selection=pre_cfg,
                        post_relax_selection=post_cfg,
                    ),
                    basin_relax_backend=IdentityRelaxBackend(),
                )
                row = {
                    "case": case_name,
                    "pre_strategy": pre_name,
                    "post_strategy": post_name,
                    "n_pose_frames": int(result.summary["n_pose_frames"]),
                    "n_pose_frames_selected_for_basin": int(result.summary["n_pose_frames_selected_for_basin"]),
                    "n_basins": int(result.summary["n_basins"]),
                    "n_nodes": int(result.summary["n_nodes"]),
                    "work_dir": run_dir.as_posix(),
                }
                rows.append(row)
    payload = {"out_root": out_root.as_posix(), "rows": rows}
    out_path = out_root / "schedule_strategy_matrix.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default="artifacts/autoresearch/schedule_strategy_matrix")
    args = p.parse_args()
    out = run_matrix(Path(args.out_root))
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
