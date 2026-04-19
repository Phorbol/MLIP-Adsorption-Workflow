from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ase.build import fcc211, molecule
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig, GeometryPairDistanceDescriptor, read_molecule_any
from adsorption_ensemble.node import NodeConfig, basin_to_node
from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder
from adsorption_ensemble.surface import SurfacePreprocessor, export_surface_detection_report


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
    pre = SurfacePreprocessor(min_surface_atoms=6)
    ctx = pre.build_context(slab)
    export_surface_detection_report(slab, ctx, out / "surface_report")
    primitives = PrimitiveBuilder().build(slab, ctx)
    sampler = PoseSampler(
        PoseSamplerConfig(
            n_rotations=2,
            n_azimuth=6,
            n_shifts=1,
            shift_radius=0.0,
            min_height=1.6,
            max_height=2.6,
            height_step=0.2,
            random_seed=0,
            max_poses_per_site=4,
        )
    )
    poses = sampler.sample(
        slab=slab,
        adsorbate=ads,
        primitives=primitives[:4],
        surface_atom_ids=ctx.detection.surface_atom_ids,
    )
    pose_frames = [slab + p.atoms for p in poses]
    if pose_frames:
        write((out / "pose_pool.extxyz").as_posix(), pose_frames)
    want_mace = bool(use_mace_dedup) and bool(mace_model_path) and Path(str(mace_model_path)).exists()
    cfg = BasinConfig(
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
    )
    basin_out = BasinBuilder(config=cfg).build(
        frames=pose_frames,
        slab_ref=slab,
        adsorbate_ref=ads,
        slab_n=len(slab),
        normal_axis=int(ctx.classification.normal_axis),
    )
    basins_frames = []
    for b in basin_out.basins:
        a = b.atoms.copy()
        a.info["basin_id"] = int(b.basin_id)
        a.info["energy_ev"] = float(b.energy_ev)
        a.info["signature"] = str(b.signature)
        basins_frames.append(a)
    if basins_frames:
        write((out / "basins.extxyz").as_posix(), basins_frames)
    (out / "basins.json").write_text(
        json.dumps(
            {
                "summary": dict(basin_out.summary),
                "relax_backend": str(basin_out.relax_backend),
                "basins": [
                    {
                        "basin_id": int(b.basin_id),
                        "energy_ev": float(b.energy_ev),
                        "denticity": int(b.denticity),
                        "signature": str(b.signature),
                        "member_candidate_ids": [int(x) for x in b.member_candidate_ids],
                        "binding_pairs": [(int(i), int(j)) for i, j in b.binding_pairs],
                    }
                    for b in basin_out.basins
                ],
                "rejected": [
                    {"candidate_id": int(r.candidate_id), "reason": str(r.reason), "metrics": dict(r.metrics)}
                    for r in basin_out.rejected
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    energy_min = basin_out.summary.get("energy_min_ev", None)
    try:
        energy_min_ev = None if energy_min is None else float(energy_min)
    except Exception:
        energy_min_ev = None
    ncfg = NodeConfig(bond_tau=1.20, node_hash_len=20)
    nodes = [basin_to_node(b, slab_n=len(slab), cfg=ncfg, energy_min_ev=energy_min_ev) for b in basin_out.basins]
    (out / "nodes.json").write_text(
        json.dumps(
            [
                {
                    "node_id": str(n.node_id),
                    "basin_id": int(n.basin_id),
                    "canonical_order": [int(x) for x in n.canonical_order],
                    "atomic_numbers": [int(x) for x in n.atomic_numbers],
                    "internal_bonds": [(int(i), int(j)) for i, j in n.internal_bonds],
                    "binding_pairs": [(int(i), int(j)) for i, j in n.binding_pairs],
                    "denticity": int(n.denticity),
                    "relative_energy_ev": (None if n.relative_energy_ev is None else float(n.relative_energy_ev)),
                    "provenance": dict(n.provenance),
                }
                for n in nodes
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": out.as_posix(),
        "n_primitives": int(len(primitives)),
        "n_poses": int(len(poses)),
        "n_basins": int(len(basin_out.basins)),
        "n_nodes": int(len(nodes)),
        "dedup_metric_requested": ("mace_node_l2" if use_mace_dedup else "rmsd"),
        "dedup_metric": str(cfg.dedup_metric),
        "files": {
            "pose_pool": (out / "pose_pool.extxyz").as_posix() if (out / "pose_pool.extxyz").exists() else "",
            "basins_extxyz": (out / "basins.extxyz").as_posix() if (out / "basins.extxyz").exists() else "",
            "basins_json": (out / "basins.json").as_posix(),
            "nodes_json": (out / "nodes.json").as_posix(),
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
