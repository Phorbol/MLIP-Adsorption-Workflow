from __future__ import annotations

from pathlib import Path
import sys

from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.conformer_md import ConformerMDSampler, ConformerMDSamplerConfig
from adsorption_ensemble.conformer_md.io_utils import read_molecule_any


def main():
    root = Path(__file__).resolve().parents[1]
    input_gjf = root / "C6.gjf"
    if not input_gjf.exists():
        raise FileNotFoundError(f"Input file not found: {input_gjf}")
    atoms = read_molecule_any(input_gjf)
    cfg = ConformerMDSamplerConfig()
    cfg.md.n_runs = 1
    cfg.selection.mode = "fps_pca_kmeans"
    cfg.selection.preselect_k = 64
    cfg.selection.pca_variance_threshold = 0.95
    cfg.selection.energy_window_ev = 0.20
    cfg.selection.rmsd_threshold = 0.05
    cfg.descriptor.backend = "mace"
    cfg.relax.backend = "mace_relax"
    cfg.descriptor.mace.model_path = "/root/.cache/mace/mace-mh-1.model"
    cfg.relax.mace.model_path = "/root/.cache/mace/mace-mh-1.model"
    cfg.descriptor.mace.head_name = "omol"
    cfg.relax.mace.head_name = "omol"
    cfg.descriptor.mace.device = "cuda"
    cfg.relax.mace.device = "cuda"
    cfg.descriptor.mace.dtype = "float32"
    cfg.relax.mace.dtype = "float32"
    cfg.descriptor.mace.max_edges_per_batch = 15000
    cfg.relax.mace.max_edges_per_batch = 15000
    cfg.relax.loose.maxf = 0.5
    cfg.relax.loose.steps = 50
    cfg.relax.refine.maxf = 0.05
    cfg.relax.refine.steps = 100
    cfg.output.work_dir = root / "artifacts" / "conformer_md"
    cfg.output.save_all_frames = True
    case_dir = cfg.output.work_dir / "c6_omol_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    write((case_dir / "C6_from_gjf.xyz").as_posix(), atoms)
    sampler = ConformerMDSampler(config=cfg)
    result = sampler.run(atoms, job_name="c6_omol_case")
    print("Conformer search finished.")
    print(f"n_selected={len(result.conformers)}")
    print(f"summary={case_dir / 'summary.txt'}")
    print(f"metadata={case_dir / 'metadata.json'}")
    print(f"ensemble={case_dir / 'ensemble.extxyz'}")


if __name__ == "__main__":
    main()
