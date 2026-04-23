from __future__ import annotations

import copy
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
    seed_mode: str = "increment_per_run"


@dataclass
class RDKitEmbedConfig:
    num_confs: int = 128
    prune_rms_thresh: float = 0.25
    embed_method: str = "etkdg_v3"
    random_seed: int = 42
    use_random_coords: bool = False
    num_threads: int = 1
    optimize_forcefield: str = "mmff"
    max_opt_iters: int = 200


@dataclass
class ConformerGeneratorConfig:
    backend: str = "xtb_md"
    xtb: XTBMDConfig = field(default_factory=XTBMDConfig)
    rdkit: RDKitEmbedConfig = field(default_factory=RDKitEmbedConfig)


@dataclass
class SelectionConfig:
    preselect_k: int = 64
    target_final_k: int | None = None
    selection_profile: str = "manual"
    mode: str = "fps_pca_kmeans"
    metric_backend: str = "auto"
    energy_window_ev: float = 0.20
    pair_energy_gap_ev: float = 0.0
    use_total_energy: bool = True
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

    @property
    def structure_metric_threshold(self) -> float:
        return float(self.rmsd_threshold)

    @structure_metric_threshold.setter
    def structure_metric_threshold(self, value: float) -> None:
        self.rmsd_threshold = float(value)


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
    generator: ConformerGeneratorConfig = field(default_factory=ConformerGeneratorConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    output: EnsembleOutputConfig = field(default_factory=EnsembleOutputConfig)

    @property
    def md(self) -> XTBMDConfig:
        return self.generator.xtb

    @md.setter
    def md(self, value: XTBMDConfig) -> None:
        self.generator.xtb = value


def resolve_selection_profile(
    config: ConformerMDSamplerConfig,
    *,
    profile: str,
    target_final_k: int | None = None,
) -> ConformerMDSamplerConfig:
    out = copy.deepcopy(config)
    key = str(profile).strip().lower()
    if key in {"manual", ""}:
        out.selection.selection_profile = "manual"
        return out
    if key == "isolated_strict":
        k_final = int(target_final_k if target_final_k is not None else (out.selection.target_final_k or 12))
        out.selection.selection_profile = "isolated_strict"
        out.selection.target_final_k = int(k_final)
        out.selection.preselect_k = int(max(64, 4 * k_final))
        out.selection.energy_window_ev = 0.20
        out.selection.pair_energy_gap_ev = 0.02
        out.selection.metric_backend = "mace"
        out.selection.use_total_energy = True
        out.descriptor.backend = "mace"
        out.descriptor.mace.dtype = "float64"
        out.descriptor.mace.enable_cueq = False
        return out
    if key == "adsorption_seed_broad":
        k_final = int(target_final_k if target_final_k is not None else (out.selection.target_final_k or 8))
        out.selection.selection_profile = "adsorption_seed_broad"
        out.selection.target_final_k = int(k_final)
        out.selection.preselect_k = int(max(96, 6 * k_final))
        out.selection.energy_window_ev = 0.60
        out.selection.pair_energy_gap_ev = 0.01
        out.selection.metric_backend = "mace"
        out.selection.use_total_energy = True
        out.descriptor.backend = "mace"
        out.descriptor.mace.dtype = "float64"
        out.descriptor.mace.enable_cueq = False
        return out
    raise ValueError(f"Unsupported selection profile: {profile}")
