from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from adsorption_ensemble.pose import PoseSweepConfig, run_pose_sampling_sweep


@dataclass
class PoseSamplingSmokeConfig:
    max_molecules: int = 2
    max_slabs: int = 2
    max_atoms_per_molecule: int = 4
    max_combinations: int = 3
    mace_disabled: bool = True


def run_pose_sampling_smoke(out_root: str | Path, cfg: PoseSweepConfig | None = None, smoke: PoseSamplingSmokeConfig | None = None) -> dict:
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    smoke_cfg = smoke or PoseSamplingSmokeConfig()
    sweep_cfg = cfg or PoseSweepConfig(
        n_rotations=2,
        n_azimuth=8,
        n_shifts=2,
        max_basis_sites=3,
        max_poses_per_site=2,
        max_poses_output=8,
        postprocess_enabled=False,
        random_seed=1,
    )
    old = os.environ.get("AE_DISABLE_MACE")
    if bool(smoke_cfg.mace_disabled):
        os.environ["AE_DISABLE_MACE"] = "1"
    try:
        out = run_pose_sampling_sweep(
            out_root=out_root_path,
            cfg=sweep_cfg,
            max_molecules=smoke_cfg.max_molecules,
            max_slabs=smoke_cfg.max_slabs,
            max_atoms_per_molecule=smoke_cfg.max_atoms_per_molecule,
            max_combinations=smoke_cfg.max_combinations,
        )
    finally:
        if old is None:
            os.environ.pop("AE_DISABLE_MACE", None)
        else:
            os.environ["AE_DISABLE_MACE"] = old
    validate_pose_sampling_run(out)
    return out


def validate_pose_sampling_run(out: dict) -> dict:
    run_dir = Path(str(out.get("run_dir", "")))
    summary_json = Path(str(out.get("summary_json", "")))
    summary_csv = Path(str(out.get("summary_csv", "")))
    report_md = Path(str(out.get("report_md", "")))
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    for p in (summary_json, summary_csv, report_md):
        if not p.exists():
            raise FileNotFoundError(f"missing output file: {p}")
    required_images = {"sites.png", "sites_only.png", "sites_inequivalent.png", "site_embedding_pca.png"}
    found = {name: 0 for name in sorted(required_images)}
    for p in run_dir.rglob("*"):
        if p.is_file() and p.name in required_images and p.stat().st_size > 0:
            found[p.name] += 1
    if any(v <= 0 for v in found.values()):
        raise RuntimeError(f"missing visualization outputs: {found}")
    rows = list(out.get("rows") or [])
    if any(bool(r.get("ok")) and bool(r.get("postprocess_enabled")) for r in rows):
        required_pp = {"pose_preselect_pca.png", "pose_loose_filter_pca.png", "pose_final_filter_pca.png"}
        ok_pp = [r for r in rows if bool(r.get("ok")) and bool(r.get("postprocess_enabled"))]
        have_any = False
        for r in ok_pp:
            od = Path(str(r.get("output_dir", "")))
            if not od.exists():
                continue
            if all((od / f).exists() and (od / f).stat().st_size > 0 for f in required_pp):
                have_any = True
                break
        if not have_any:
            raise RuntimeError("postprocess_enabled but PCA comparison plots are missing in all ok cases.")
    if any(bool(r.get("ok")) and bool(r.get("ensemble_enabled")) for r in rows):
        ok_e = [r for r in rows if bool(r.get("ok")) and bool(r.get("ensemble_enabled"))]
        have_any = False
        for r in ok_e:
            od = Path(str(r.get("output_dir", "")))
            if not od.exists():
                continue
            need = [od / "basins.json", od / "nodes.json"]
            if all(p.exists() and p.stat().st_size > 0 for p in need):
                have_any = True
                break
        if not have_any:
            raise RuntimeError("ensemble_enabled but basins.json/nodes.json are missing in all ok cases.")
    return {"run_dir": run_dir.as_posix(), "found": found}
