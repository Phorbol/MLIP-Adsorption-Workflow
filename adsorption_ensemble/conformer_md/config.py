from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class XTBMDConfig:
    temperature_k: float = 400.0
    time_ps: float = 100.0
    step_fs: float = 1.0
    dump_fs: float = 50.0
    seed: int = 42
    hmass: int = 1
    shake: int = 1
    xtb_executable: str = "xtb"
    gfnff: bool = True
    n_runs: int = 1


@dataclass
class SelectionConfig:
    preselect_k: int = 64
    mode: str = "fps_pca_kmeans"
    energy_window_ev: float = 0.20
    rmsd_threshold: float = 0.05
    loose_filter: str = "dual"
    final_filter: str = "dual"
    loose_energy_window_ev: float | None = None
    loose_rmsd_threshold: float | None = None
    final_energy_window_ev: float | None = None
    final_rmsd_threshold: float | None = None
    fps_seed: int = 0
    pca_variance_threshold: float = 0.95
    fps_pool_factor: int = 3
    fps_seed_indices: tuple[int, ...] = ()
    fps_round_size: int | None = None
    fps_rounds: int | None = None
    fps_convergence_enable: bool = False
    fps_convergence_pca_var: float = 0.95
    fps_convergence_grid_bins: int = 12
    fps_convergence_min_rounds: int = 5
    fps_convergence_patience: int = 3
    fps_convergence_min_coverage_gain: float = 1e-3
    fps_convergence_min_novelty: float = 5e-2


@dataclass
class MACEInferenceConfig:
    model_path: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    enable_cueq: bool = False
    max_edges_per_batch: int = 15000
    num_workers: int = 1
    layers_to_keep: int = -1
    mlp_energy_key: str | None = None
    head_name: str = "Default"


@dataclass
class DescriptorConfig:
    backend: str = "geometry"
    use_float64: bool = False
    mace: MACEInferenceConfig = field(default_factory=MACEInferenceConfig)


@dataclass
class RelaxStageConfig:
    maxf: float = 0.5
    steps: int = 50


@dataclass
class RelaxConfig:
    backend: str = "mace_relax"
    mace: MACEInferenceConfig = field(default_factory=MACEInferenceConfig)
    loose: RelaxStageConfig = field(default_factory=RelaxStageConfig)
    refine: RelaxStageConfig = field(default_factory=lambda: RelaxStageConfig(maxf=0.05, steps=100))


@dataclass
class EnsembleOutputConfig:
    save_all_frames: bool = False
    work_dir: Path = field(default_factory=lambda: Path("artifacts") / "conformer_md")


@dataclass
class ConformerMDSamplerConfig:
    md: XTBMDConfig = field(default_factory=XTBMDConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    output: EnsembleOutputConfig = field(default_factory=EnsembleOutputConfig)
