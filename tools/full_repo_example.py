from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ase.build import fcc211, molecule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin import BasinConfig
from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, GeometryPairDistanceDescriptor, read_molecule_any
from adsorption_ensemble.node import NodeConfig
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, evaluate_adsorption_workflow_readiness, run_adsorption_workflow


class FakeMDRunner:
    def __init__(self, n_frames: int = 40):
        self.n_frames = int(n_frames)

    def run(self, molecule_atoms, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(self.n_frames):
            a = molecule_atoms.copy()
            shift = 0.03 * np.sin(0.3 * i + np.arange(len(a))[:, None])
            a.set_positions(a.get_positions() + shift)
            frames.append(a)
        return type("MDRunResult", (), {"frames": frames, "metadata": {"source": "fake", "n_frames": len(frames)}})()


class FakeRelaxBackend:
    def relax_batch(self, frames, work_dir: Path, maxf=None, steps=None):
        work_dir.mkdir(parents=True, exist_ok=True)
        out = [f.copy() for f in frames]
        energies = np.linspace(0.0, 0.22, len(out), dtype=float)
        return out, energies


def run_adsorption_api_example(
    out_root: Path,
    use_mace_dedup: bool,
    mace_model_path: str | None,
    mace_device: str,
    mace_dtype: str,
) -> dict:
    out = out_root / "adsorption_api"
    out.mkdir(parents=True, exist_ok=True)
    slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
    ads = molecule("CO")
    want_mace = bool(use_mace_dedup) and bool(mace_model_path) and Path(str(mace_model_path)).exists()
    cfg = AdsorptionWorkflowConfig(
        work_dir=out,
        pose_sampler_config=PoseSamplerConfig(
            n_rotations=2,
            n_azimuth=6,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.6,
            max_height=2.6,
            height_step=0.2,
            random_seed=0,
            max_poses_per_site=4,
        ),
        basin_config=BasinConfig(
            relax_maxf=0.10,
            relax_steps=2,
            energy_window_ev=1.0,
            dedup_metric=("mace_node_l2" if want_mace else "rmsd"),
            rmsd_threshold=0.10,
            mace_node_l2_threshold=2.0,
            mace_model_path=(str(mace_model_path) if want_mace else None),
            mace_device=str(mace_device),
            mace_dtype=str(mace_dtype),
            binding_tau=1.15,
            desorption_min_bonds=0,
            work_dir=out / "basin_work",
        ),
        node_config=NodeConfig(bond_tau=1.20, node_hash_len=20),
        max_primitives=4,
    )
    result = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
    readiness = evaluate_adsorption_workflow_readiness(result)
    return {
        "out_dir": out.as_posix(),
        "n_primitives": int(result.summary["n_primitives"]),
        "n_poses": int(result.summary["n_pose_frames"]),
        "n_basins": int(result.summary["n_basins"]),
        "n_nodes": int(result.summary["n_nodes"]),
        "dedup_metric_requested": ("mace_node_l2" if use_mace_dedup else "rmsd"),
        "dedup_metric": str(cfg.basin_config.dedup_metric),
        "paper_readiness_score": int(readiness.score),
        "paper_readiness_max_score": int(readiness.max_score),
        "files": {
            "pose_pool": result.artifacts.get("pose_pool_extxyz", ""),
            "basins_extxyz": result.artifacts.get("basins_extxyz", ""),
            "basins_json": result.artifacts.get("basins_json", ""),
            "nodes_json": result.artifacts.get("nodes_json", ""),
            "site_dictionary_json": result.artifacts.get("site_dictionary_json", ""),
            "workflow_summary_json": result.artifacts.get("workflow_summary_json", ""),
        },
    }


def run_conformer_md_example(out_root: Path) -> dict:
    out = out_root / "conformer_md"
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    mol_path = root / "C6.gjf"
    mol = read_molecule_any(mol_path) if mol_path.exists() else molecule("H2O")
    cfg = ConformerMDSamplerConfig()
    cfg.output.work_dir = out
    cfg.selection.preselect_k = 12
    cfg.selection.mode = "fps"
    cfg.selection.energy_window_ev = 0.20
    cfg.selection.rmsd_threshold = 0.02
    sampler = ConformerMDSampler(
        config=cfg,
        md_runner=FakeMDRunner(n_frames=60),
        descriptor_extractor=GeometryPairDistanceDescriptor(),
        relax_backend=FakeRelaxBackend(),
    )
    result = sampler.run(mol, job_name="example")
    return {
        "out_dir": (out / "example").as_posix(),
        "n_conformers": int(len(result.conformers)),
        "top5_energy_ev": [float(x) for x in np.asarray(result.energies_ev, dtype=float)[:5].tolist()],
        "files": {
            "ensemble_extxyz": (out / "example" / "ensemble.extxyz").as_posix(),
            "summary_json": (out / "example" / "summary.json").as_posix(),
            "summary_txt": (out / "example" / "summary.txt").as_posix(),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default="artifacts/full_repo_example")
    p.add_argument("--use-mace-dedup", action="store_true")
    p.add_argument("--mace-model-path", type=str, default="")
    p.add_argument("--mace-device", type=str, default="cuda")
    p.add_argument("--mace-dtype", type=str, default="float64")
    p.add_argument("--skip-conformer-md", action="store_true")
    args = p.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"out_root": out_root.as_posix()}
    payload["adsorption_api"] = run_adsorption_api_example(
        out_root=out_root,
        use_mace_dedup=bool(args.use_mace_dedup),
        mace_model_path=(str(args.mace_model_path).strip() if str(args.mace_model_path).strip() else None),
        mace_device=str(args.mace_device),
        mace_dtype=str(args.mace_dtype),
    )
    if not bool(args.skip_conformer_md):
        payload["conformer_md"] = run_conformer_md_example(out_root=out_root)
    (out_root / "example_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
