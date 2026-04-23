from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from ase import Atoms
from ase.io import write
from adsorption_ensemble.selection.strategies import RMSDSelector

from .config import ConformerMDSamplerConfig
from .descriptors import DescriptorExtractor, GeometryPairDistanceDescriptor, MACEInvariantDescriptor
from .mace_inference import MACEBatchInferencer
from .rdkit_generator import RDKitConformerGenerator
from .selectors import ConformerSelector
from .xtb import MDRunResult, XTBMDRunner


class ConformerGenerator(Protocol):
    def generate(self, molecule: Atoms, run_dir: Path) -> MDRunResult:
        ...


class RelaxBackend(Protocol):
    def relax_batch(
        self,
        frames: list[Atoms],
        work_dir: Path,
        maxf: float | None = None,
        steps: int | None = None,
    ) -> tuple[list[Atoms], np.ndarray]:
        ...


@dataclass
class ConformerEnsemble:
    conformers: list[Atoms]
    energies_ev: np.ndarray
    selected_ids: list[int]
    metadata: dict


class IdentityRelaxBackend:
    def relax_batch(
        self,
        frames: list[Atoms],
        work_dir: Path,
        maxf: float | None = None,
        steps: int | None = None,
    ) -> tuple[list[Atoms], np.ndarray]:
        energies = np.zeros(len(frames), dtype=float)
        return frames, energies


class MACEEnergyRelaxBackend:
    def __init__(self, inferencer: MACEBatchInferencer):
        self.inferencer = inferencer
        self.last_infer_metadata: dict | None = None

    def relax_batch(
        self,
        frames: list[Atoms],
        work_dir: Path,
        maxf: float | None = None,
        steps: int | None = None,
    ) -> tuple[list[Atoms], np.ndarray]:
        out = self.inferencer.infer(frames)
        self.last_infer_metadata = dict(out.metadata)
        return frames, np.asarray(out.energies_per_atom_ev, dtype=float)


class MACERelaxBackend:
    def __init__(self, inferencer: MACEBatchInferencer):
        self.inferencer = inferencer
        self.last_infer_metadata: dict | None = None

    def relax_batch(
        self,
        frames: list[Atoms],
        work_dir: Path,
        maxf: float | None = None,
        steps: int | None = None,
    ) -> tuple[list[Atoms], np.ndarray]:
        from mace.calculators import MACECalculator
        from .mace_batch_relax import BatchRelaxer, resolve_runtime_device

        work_dir.mkdir(parents=True, exist_ok=True)
        runtime_device, runtime_meta = resolve_runtime_device(self.inferencer.config.device)
        rank_suffix = f".rank{runtime_meta['rank']}" if runtime_meta["world_size"] > 1 else ""
        calc_kwargs = {
            "model_paths": [self.inferencer.config.model_path],
            "device": runtime_device,
            "default_dtype": self.inferencer.config.dtype,
        }
        try:
            calculator = MACECalculator(head=self.inferencer.config.head_name, **calc_kwargs)
        except TypeError:
            calculator = MACECalculator(**calc_kwargs)
        relaxer = BatchRelaxer(
            calculator=calculator,
            max_edges_per_batch=self.inferencer.config.max_edges_per_batch,
            device=runtime_device,
        )
        relaxed_raw = relaxer.relax(
            atoms_list=[a.copy() for a in frames],
            fmax=float(maxf if maxf is not None else 0.05),
            head=self.inferencer.config.head_name,
            max_n_steps=int(steps if steps is not None else 100),
            inplace=True,
            trajectory_dir=(work_dir / "traj").as_posix(),
            append_trajectory_file=(work_dir / f"relaxed_stream{rank_suffix}.extxyz").as_posix(),
            save_log_file=(work_dir / f"batch_relax{rank_suffix}.log").as_posix(),
        )
        out_frames: list[Atoms] = []
        n_ok = 0
        n_fail = 0
        energies_epa: list[float] = []
        for i, relaxed in enumerate(relaxed_raw):
            if relaxed is None:
                out_frames.append(frames[i].copy())
                n_fail += 1
                energies_epa.append(float("nan"))
            else:
                out_frames.append(relaxed)
                n_ok += 1
                try:
                    energies_epa.append(float(relaxed.get_potential_energy()) / float(len(relaxed)))
                except Exception:
                    energies_epa.append(float("nan"))
        meta = {
            "relax_backend": "mace_relax",
            "maxf": float(maxf) if maxf is not None else None,
            "steps": int(steps) if steps is not None else None,
            "n_relaxed_ok": int(n_ok),
            "n_relaxed_failed": int(n_fail),
            "runtime_rank": runtime_meta["rank"],
            "runtime_world_size": runtime_meta["world_size"],
            "runtime_local_rank": runtime_meta["local_rank"],
            "runtime_device": runtime_meta["device"],
        }
        meta.update(
            {"energies_from_calc": True, "n_energy_nan": int(np.sum(~np.isfinite(np.asarray(energies_epa, dtype=float))))}
        )
        self.last_infer_metadata = meta
        return out_frames, np.asarray(energies_epa, dtype=float)


class ConformerMDSampler:
    def __init__(
        self,
        config: ConformerMDSamplerConfig,
        md_runner: object | None = None,
        descriptor_extractor: DescriptorExtractor | None = None,
        relax_backend: RelaxBackend | None = None,
    ):
        self.config = config
        self.conformer_generator = md_runner or self._build_conformer_generator(config)
        if descriptor_extractor is None:
            self.descriptor_extractor = self._build_descriptor_extractor(config)
        else:
            self.descriptor_extractor = descriptor_extractor
        if relax_backend is None:
            self.relax_backend = self._build_relax_backend(config)
        else:
            self.relax_backend = relax_backend
        self.selector = ConformerSelector(config.selection)

    def run(self, molecule: Atoms, job_name: str = "conformer_job") -> ConformerEnsemble:
        run_dir = self.config.output.work_dir / job_name
        run_dir.mkdir(parents=True, exist_ok=True)
        all_frames, run_metadata, generator_summary = self._generate_raw_frames(molecule=molecule, run_dir=run_dir)
        if not all_frames:
            raise RuntimeError("No conformer frames were produced.")
        return self.run_from_frames(
            frames=all_frames,
            job_name=job_name,
            md_runs_metadata=run_metadata,
            raw_features=None,
            generator_summary=generator_summary,
        )

    def run_from_frames(
        self,
        frames: list[Atoms],
        job_name: str,
        md_runs_metadata: list[dict] | None = None,
        raw_features: np.ndarray | None = None,
        generator_summary: dict | None = None,
    ) -> ConformerEnsemble:
        run_dir = self.config.output.work_dir / job_name
        run_dir.mkdir(parents=True, exist_ok=True)
        all_frames = frames
        if not all_frames:
            raise RuntimeError("No frames were provided.")
        if raw_features is None:
            raw_features = self.descriptor_extractor.transform(all_frames)
        raw_energies = self._resolve_raw_energies(all_frames)
        pre_ids = self.selector._preselect(energies=raw_energies, features=np.asarray(raw_features, dtype=float))
        pre_frames = [all_frames[i] for i in pre_ids]
        metric_extractor = self._build_metric_extractor(self.config)
        pre_features = metric_extractor.transform(pre_frames)
        loose_frames, loose_energies = self.relax_backend.relax_batch(
            pre_frames,
            run_dir / "relax_loose",
            maxf=self.config.relax.loose.maxf,
            steps=self.config.relax.loose.steps,
        )
        loose_energies = self._apply_energy_semantics(loose_energies, loose_frames)
        if len(loose_frames) != len(pre_ids) or len(loose_energies) != len(pre_ids):
            raise ValueError("Loose relax backend output shape mismatch.")
        loose_features = metric_extractor.transform(loose_frames)
        loose_keep_ids = self._apply_filter(
            energies=np.asarray(loose_energies, dtype=float),
            features=np.asarray(loose_features, dtype=float),
            energy_window_ev=self.config.selection.loose_energy_window_ev,
            rmsd_threshold=self.config.selection.loose_rmsd_threshold,
            strategy=self.config.selection.loose_filter,
        )
        loose_keep_frames = [loose_frames[i] for i in loose_keep_ids]
        loose_keep_features = np.asarray([loose_features[i] for i in loose_keep_ids], dtype=float) if loose_keep_ids else np.empty((0, 0), dtype=float)
        loose_keep_energies = np.asarray([loose_energies[i] for i in loose_keep_ids], dtype=float) if loose_keep_ids else np.empty((0,), dtype=float)
        refine_input = [loose_frames[i] for i in loose_keep_ids]
        refine_frames, refine_energies = self.relax_backend.relax_batch(
            refine_input,
            run_dir / "relax_refine",
            maxf=self.config.relax.refine.maxf,
            steps=self.config.relax.refine.steps,
        )
        refine_energies = self._apply_energy_semantics(refine_energies, refine_frames)
        if len(refine_frames) != len(refine_input) or len(refine_energies) != len(refine_input):
            raise ValueError("Refine relax backend output shape mismatch.")
        refine_features = metric_extractor.transform(refine_frames)
        final_ids = self._apply_filter(
            energies=np.asarray(refine_energies, dtype=float),
            features=np.asarray(refine_features, dtype=float),
            energy_window_ev=self.config.selection.final_energy_window_ev,
            rmsd_threshold=self.config.selection.final_rmsd_threshold,
            strategy=self.config.selection.final_filter,
        )
        final_ids = self._apply_target_final_k(
            candidate_ids=final_ids,
            energies=np.asarray(refine_energies, dtype=float),
            features=np.asarray(refine_features, dtype=float),
        )
        selected_relaxed = [refine_frames[i] for i in final_ids]
        selected_energies = np.asarray([refine_energies[i] for i in final_ids], dtype=float)
        selected_features = np.asarray([refine_features[i] for i in final_ids], dtype=float) if final_ids else np.empty((0, 0), dtype=float)
        stage_metrics = self._build_stage_metrics(
            all_frames=all_frames,
            pre_frames=pre_frames,
            pre_features=pre_features,
            loose_frames=loose_frames,
            loose_energies=np.asarray(loose_energies, dtype=float),
            loose_features=np.asarray(loose_features, dtype=float),
            loose_keep_frames=loose_keep_frames,
            loose_keep_energies=loose_keep_energies,
            loose_keep_features=loose_keep_features,
            refine_frames=refine_frames,
            refine_energies=np.asarray(refine_energies, dtype=float),
            refine_features=np.asarray(refine_features, dtype=float),
            selected_frames=selected_relaxed,
            selected_energies=selected_energies,
            selected_features=selected_features,
        )
        result_summary = self._build_result_summary(selected_energies=selected_energies)
        metadata = {
            "job_name": job_name,
            "config": asdict(self.config),
            "generator_backend": str(self._generator_backend(self.config)),
            "generator_summary": dict(generator_summary or {}),
            "generator_runs": [dict(x) for x in (md_runs_metadata or [])],
            "selection_profile": str(self.config.selection.selection_profile),
            "resolved_preselect_k": int(self.config.selection.preselect_k),
            "resolved_target_final_k": (
                None if self.config.selection.target_final_k is None else int(self.config.selection.target_final_k)
            ),
            "resolved_metric_backend": str(self._resolve_metric_backend(self.config)),
            "resolved_structure_metric_threshold": float(self.config.selection.rmsd_threshold),
            "resolved_energy_window_ev": float(self.config.selection.energy_window_ev),
            "resolved_pair_energy_gap_ev": float(self.config.selection.pair_energy_gap_ev),
            "energy_semantics": str(self._energy_semantics_label()),
            "use_total_energy": bool(self.config.selection.use_total_energy),
            "md_runs": ([dict(x) for x in (md_runs_metadata or [])] if self._generator_backend(self.config) == "xtb_md" else []),
            "per_run_seeds": [int(m["seed"]) for m in (md_runs_metadata or []) if "seed" in m],
            "n_raw_frames": len(all_frames),
            "n_preselected": len(pre_ids),
            "n_after_loose_filter": len(loose_keep_ids),
            "n_after_refine": len(refine_frames),
            "n_selected": len(final_ids),
            "preselected_ids_in_raw": pre_ids,
            "selected_ids_in_refine": final_ids,
            "raw_energy_source": str(self._raw_energy_source(all_frames)),
            "result_summary": result_summary,
            "stage_metrics": stage_metrics,
        }
        descriptor_meta = getattr(self.descriptor_extractor, "last_infer_metadata", None)
        if descriptor_meta is not None:
            metadata["descriptor_inference"] = dict(descriptor_meta)
        relax_meta = getattr(self.relax_backend, "last_infer_metadata", None)
        if relax_meta is not None:
            metadata["relax_inference"] = dict(relax_meta)
        self._write_outputs(
            run_dir=run_dir,
            all_frames=all_frames,
            pre_frames=pre_frames,
            loose_frames=loose_frames,
            loose_keep_frames=loose_keep_frames,
            refine_frames=refine_frames,
            selected=selected_relaxed,
            selected_energies=selected_energies,
            metadata=metadata,
        )
        return ConformerEnsemble(
            conformers=selected_relaxed,
            energies_ev=selected_energies,
            selected_ids=final_ids,
            metadata=metadata,
        )

    def _apply_filter(
        self,
        energies: np.ndarray,
        features: np.ndarray,
        energy_window_ev: float | None,
        rmsd_threshold: float | None,
        strategy: str,
    ) -> list[int]:
        from adsorption_ensemble.selection.strategies import DualThresholdSelector, EnergyWindowFilter, RMSDSelector

        if energy_window_ev is None:
            energy_window_ev = float(self.config.selection.energy_window_ev)
        if rmsd_threshold is None:
            rmsd_threshold = float(self.config.selection.rmsd_threshold)
        cand = list(range(len(features)))
        strat = str(strategy).lower()
        if strat == "none":
            return sorted(cand, key=lambda i: float(energies[i]))
        if strat == "energy":
            kept = EnergyWindowFilter(delta_e=float(energy_window_ev)).select(energies=energies, candidate_ids=cand)
            return sorted(kept, key=lambda i: float(energies[i]))
        if strat == "rmsd":
            ordered = sorted(cand, key=lambda i: float(energies[i]))
            return self._select_by_metric_threshold_with_pair_gap(
                energies=energies,
                features=features,
                candidate_ids=ordered,
                metric_threshold=float(rmsd_threshold),
            )
        dual = DualThresholdSelector(
            energy_window=EnergyWindowFilter(delta_e=float(energy_window_ev)),
            rmsd_selector=RMSDSelector(threshold=float(rmsd_threshold)),
        )
        kept = dual.energy_window.select(energies=energies, candidate_ids=cand)
        ordered = sorted(kept, key=lambda i: float(energies[i]))
        return self._select_by_metric_threshold_with_pair_gap(
            energies=energies,
            features=features,
            candidate_ids=ordered,
            metric_threshold=float(rmsd_threshold),
        )

    def _write_outputs(
        self,
        run_dir: Path,
        all_frames: list[Atoms],
        pre_frames: list[Atoms],
        loose_frames: list[Atoms],
        loose_keep_frames: list[Atoms],
        refine_frames: list[Atoms],
        selected: list[Atoms],
        selected_energies: np.ndarray,
        metadata: dict,
    ) -> None:
        if all_frames:
            write((run_dir / "raw_generated.extxyz").as_posix(), all_frames)
        if self.config.output.save_all_frames:
            write((run_dir / "all_frames.extxyz").as_posix(), all_frames)
        if pre_frames:
            write((run_dir / "preselected.extxyz").as_posix(), pre_frames)
        if loose_frames:
            write((run_dir / "loose_relaxed.extxyz").as_posix(), loose_frames)
        if loose_keep_frames:
            write((run_dir / "loose_filtered.extxyz").as_posix(), loose_keep_frames)
        if refine_frames:
            write((run_dir / "refined.extxyz").as_posix(), refine_frames)
        if selected:
            for i, atoms in enumerate(selected):
                atoms.info["energy_ev"] = float(selected_energies[i])
            write((run_dir / "ensemble.extxyz").as_posix(), selected)
        meta_json = run_dir / "metadata.json"
        meta_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        summary_json = run_dir / "summary.json"
        summary_json.write_text(json.dumps(metadata.get("result_summary", {}), indent=2, default=str), encoding="utf-8")
        stage_metrics = metadata.get("stage_metrics", {})
        (run_dir / "stage_metrics.json").write_text(json.dumps(stage_metrics, indent=2, default=str), encoding="utf-8")
        self._write_stage_metrics_csv(run_dir=run_dir, stage_metrics=stage_metrics)
        self._write_summary_text(run_dir=run_dir, metadata=metadata)

    @staticmethod
    def _build_descriptor_extractor(config: ConformerMDSamplerConfig) -> DescriptorExtractor:
        backend = config.descriptor.backend.lower()
        if backend == "geometry":
            return GeometryPairDistanceDescriptor(use_float64=config.descriptor.use_float64)
        if backend == "mace":
            infer = MACEBatchInferencer(config.descriptor.mace)
            return MACEInvariantDescriptor(inferencer=infer)
        raise ValueError(f"Unsupported descriptor backend: {config.descriptor.backend}")

    @staticmethod
    def _build_metric_extractor(config: ConformerMDSamplerConfig) -> DescriptorExtractor:
        backend = ConformerMDSampler._resolve_metric_backend(config)
        if backend == "geometry":
            return GeometryPairDistanceDescriptor(use_float64=config.descriptor.use_float64)
        if backend == "mace":
            mace_cfg = copy.deepcopy(config.descriptor.mace)
            infer = MACEBatchInferencer(mace_cfg)
            return MACEInvariantDescriptor(inferencer=infer)
        raise ValueError(f"Unsupported conformer metric backend: {config.selection.metric_backend}")

    @staticmethod
    def _resolve_metric_backend(config: ConformerMDSamplerConfig) -> str:
        backend = str(config.selection.metric_backend).strip().lower()
        if backend in {"", "auto"}:
            backend = config.descriptor.backend.lower()
        return backend

    @staticmethod
    def _generator_backend(config: ConformerMDSamplerConfig) -> str:
        return str(config.generator.backend).strip().lower()

    @staticmethod
    def _build_conformer_generator(config: ConformerMDSamplerConfig):
        backend = ConformerMDSampler._generator_backend(config)
        if backend == "xtb_md":
            return XTBMDRunner(config.generator.xtb)
        if backend == "rdkit_embed":
            return RDKitConformerGenerator(config.generator.rdkit)
        raise ValueError(f"Unsupported conformer generator backend: {config.generator.backend}")

    @staticmethod
    def _build_relax_backend(config: ConformerMDSamplerConfig) -> RelaxBackend:
        backend = config.relax.backend.lower()
        if backend == "identity":
            return IdentityRelaxBackend()
        if backend == "mace_energy":
            infer = MACEBatchInferencer(config.relax.mace)
            return MACEEnergyRelaxBackend(inferencer=infer)
        if backend == "mace_relax":
            infer = MACEBatchInferencer(config.relax.mace)
            return MACERelaxBackend(inferencer=infer)
        raise ValueError(f"Unsupported relax backend: {config.relax.backend}")

    def _apply_md_run_seed(self, run_index: int) -> None:
        mode = str(getattr(self.config.md, "seed_mode", "increment_per_run")).strip().lower()
        seed0 = int(self.config.md.seed)
        if mode == "fixed":
            seed_use = seed0
        elif mode == "increment_per_run":
            seed_use = seed0 + int(run_index)
        elif mode == "hashed":
            seed_use = abs(hash((seed0, int(run_index)))) % (2**31 - 1)
        else:
            raise ValueError(f"Unsupported MD seed mode: {self.config.md.seed_mode}")
        if hasattr(self.conformer_generator, "config"):
            try:
                self.conformer_generator.config.seed = int(seed_use)
            except Exception:
                pass

    def _generate_once(self, molecule: Atoms, run_dir: Path) -> MDRunResult:
        if hasattr(self.conformer_generator, "generate"):
            return self.conformer_generator.generate(molecule, run_dir)
        if hasattr(self.conformer_generator, "run"):
            return self.conformer_generator.run(molecule, run_dir)
        raise TypeError("Conformer generator must expose generate(...) or run(...).")

    def _generate_raw_frames(self, molecule: Atoms, run_dir: Path) -> tuple[list[Atoms], list[dict], dict]:
        backend = self._generator_backend(self.config)
        all_frames: list[Atoms] = []
        run_metadata: list[dict] = []
        t0 = time.perf_counter()
        if backend == "xtb_md":
            for i in range(max(1, self.config.md.n_runs)):
                child = run_dir / f"md_run_{i:03d}"
                self._apply_md_run_seed(run_index=i)
                result = self._generate_once(molecule, child)
                all_frames.extend(result.frames)
                run_metadata.append(dict(result.metadata))
            summary = {
                "generator_backend": "xtb_md",
                "n_runs": int(len(run_metadata)),
                "n_raw_frames": int(len(all_frames)),
                "walltime_generation_s": float(time.perf_counter() - t0),
            }
            return all_frames, run_metadata, summary
        if backend == "rdkit_embed":
            child = run_dir / "rdkit_embed"
            result = self._generate_once(molecule, child)
            all_frames = [a.copy() for a in result.frames]
            run_metadata = [dict(result.metadata)]
            summary = dict(result.metadata)
            summary["n_raw_frames"] = int(len(all_frames))
            summary["walltime_generation_s"] = float(time.perf_counter() - t0)
            return all_frames, run_metadata, summary
        raise ValueError(f"Unsupported conformer generator backend: {self.config.generator.backend}")

    def _resolve_raw_energies(self, frames: list[Atoms]) -> np.ndarray:
        supplied = getattr(self.descriptor_extractor, "last_energies_ev", None)
        if supplied is not None:
            arr = np.asarray(supplied, dtype=float)
            if len(arr) == len(frames):
                return self._apply_energy_semantics(arr, frames)
        frame_info_vals = [a.info.get("generator_energy_ev", None) for a in frames]
        if frame_info_vals and all(v is not None for v in frame_info_vals):
            try:
                return self._apply_energy_semantics(np.asarray(frame_info_vals, dtype=float), frames)
            except Exception:
                pass
        return self._apply_energy_semantics(np.zeros(len(frames), dtype=float), frames)

    def _raw_energy_source(self, frames: list[Atoms] | None = None) -> str:
        supplied = getattr(self.descriptor_extractor, "last_energies_ev", None)
        if supplied is not None:
            return "descriptor"
        if frames is not None:
            vals = [a.info.get("generator_energy_ev", None) for a in frames]
            if vals and all(v is not None for v in vals):
                return "frame_info"
        return "zeros"

    def _apply_energy_semantics(self, energies: np.ndarray, frames: list[Atoms]) -> np.ndarray:
        arr = np.asarray(energies, dtype=float)
        if not bool(self.config.selection.use_total_energy):
            return arr
        if len(arr) != len(frames):
            return arr
        counts = np.asarray([len(a) for a in frames], dtype=float)
        return arr * counts

    def _energy_semantics_label(self) -> str:
        return "total_ev" if bool(self.config.selection.use_total_energy) else "per_atom_ev"

    def _apply_target_final_k(self, candidate_ids: list[int], energies: np.ndarray, features: np.ndarray) -> list[int]:
        k = self.config.selection.target_final_k
        if k is None or int(k) <= 0 or len(candidate_ids) <= int(k):
            return sorted(candidate_ids, key=lambda i: float(energies[i]))
        ordered = sorted(candidate_ids, key=lambda i: float(energies[i]))
        if features.ndim != 2 or features.shape[0] == 0:
            return ordered[: int(k)]
        selected = [ordered[0]]
        remaining = ordered[1:]
        while remaining and len(selected) < int(k):
            best_id = remaining[0]
            best_score = None
            for idx in remaining:
                ref = features[selected]
                diff = ref - features[idx]
                min_dist = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
                pair_gap = min(abs(float(energies[idx]) - float(energies[j])) for j in selected)
                score = (min_dist, pair_gap, -float(energies[idx]))
                if best_score is None or score > best_score:
                    best_score = score
                    best_id = idx
            selected.append(best_id)
            remaining = [idx for idx in remaining if idx != best_id]
        return selected

    def _select_by_metric_threshold_with_pair_gap(
        self,
        energies: np.ndarray,
        features: np.ndarray,
        candidate_ids: list[int],
        metric_threshold: float,
    ) -> list[int]:
        pair_gap = float(self.config.selection.pair_energy_gap_ev)
        if pair_gap <= 0.0:
            return RMSDSelector(threshold=float(metric_threshold)).select(features=features, candidate_ids=candidate_ids)
        selected: list[int] = []
        for idx in candidate_ids:
            if not selected:
                selected.append(int(idx))
                continue
            ref = features[selected]
            diff = ref - features[idx]
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            close_mask = dists < float(metric_threshold)
            if not np.any(close_mask):
                selected.append(int(idx))
                continue
            close_selected = [selected[pos] for pos, flag in enumerate(close_mask.tolist()) if bool(flag)]
            if all(abs(float(energies[idx]) - float(energies[j])) >= pair_gap for j in close_selected):
                selected.append(int(idx))
        return selected

    @staticmethod
    def _build_result_summary(selected_energies: np.ndarray) -> dict:
        e = np.asarray(selected_energies, dtype=float)
        if len(e) == 0:
            return {
                "n_selected": 0,
                "energy_min_ev": None,
                "energy_max_ev": None,
                "energy_mean_ev": None,
                "energy_std_ev": None,
                "top5_energy_ev": [],
            }
        e_sorted = np.sort(e)
        return {
            "n_selected": int(len(e)),
            "energy_min_ev": float(np.min(e)),
            "energy_max_ev": float(np.max(e)),
            "energy_mean_ev": float(np.mean(e)),
            "energy_std_ev": float(np.std(e)),
            "top5_energy_ev": [float(x) for x in e_sorted[:5]],
        }

    @staticmethod
    def _write_summary_text(run_dir: Path, metadata: dict) -> None:
        summary = metadata.get("result_summary", {})
        lines = [
            f"job_name: {metadata.get('job_name', '')}",
            f"generator_backend: {metadata.get('generator_backend', '')}",
            f"selection_profile: {metadata.get('selection_profile', '')}",
            f"energy_semantics: {metadata.get('energy_semantics', '')}",
            f"target_final_k: {metadata.get('resolved_target_final_k')}",
            f"n_raw_frames: {metadata.get('n_raw_frames', 0)}",
            f"n_preselected: {metadata.get('n_preselected', 0)}",
            f"n_selected: {metadata.get('n_selected', 0)}",
            f"energy_min_ev: {summary.get('energy_min_ev')}",
            f"energy_max_ev: {summary.get('energy_max_ev')}",
            f"energy_mean_ev: {summary.get('energy_mean_ev')}",
            f"energy_std_ev: {summary.get('energy_std_ev')}",
            f"top5_energy_ev: {summary.get('top5_energy_ev', [])}",
        ]
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _build_stage_metrics(
        all_frames: list[Atoms],
        pre_frames: list[Atoms],
        pre_features: np.ndarray,
        loose_frames: list[Atoms],
        loose_energies: np.ndarray,
        loose_features: np.ndarray,
        loose_keep_frames: list[Atoms],
        loose_keep_energies: np.ndarray,
        loose_keep_features: np.ndarray,
        refine_frames: list[Atoms],
        refine_energies: np.ndarray,
        refine_features: np.ndarray,
        selected_frames: list[Atoms],
        selected_energies: np.ndarray,
        selected_features: np.ndarray,
    ) -> dict:
        n_raw = len(all_frames)
        n_pre = len(pre_frames)
        n_loose = len(loose_frames)
        n_loose_keep = len(loose_keep_frames)
        n_refine = len(refine_frames)
        n_final = len(selected_frames)
        return {
            "counts": {
                "raw": n_raw,
                "preselected": n_pre,
                "loose_relaxed": n_loose,
                "loose_filtered": n_loose_keep,
                "refined": n_refine,
                "final": n_final,
            },
            "retention": {
                "pre_over_raw": ConformerMDSampler._safe_ratio(n_pre, n_raw),
                "loose_filtered_over_loose": ConformerMDSampler._safe_ratio(n_loose_keep, n_loose),
                "final_over_refine": ConformerMDSampler._safe_ratio(n_final, n_refine),
                "final_over_raw": ConformerMDSampler._safe_ratio(n_final, n_raw),
            },
            "dedup_removed": {
                "loose_filter_removed": int(max(0, n_loose - n_loose_keep)),
                "final_filter_removed": int(max(0, n_refine - n_final)),
            },
            "energy": {
                "loose_relaxed": ConformerMDSampler._energy_stats(loose_energies),
                "loose_filtered": ConformerMDSampler._energy_stats(loose_keep_energies),
                "refined": ConformerMDSampler._energy_stats(refine_energies),
                "final": ConformerMDSampler._energy_stats(selected_energies),
            },
            "diversity": {
                "preselected": ConformerMDSampler._pair_distance_stats(pre_features),
                "loose_relaxed": ConformerMDSampler._pair_distance_stats(loose_features),
                "loose_filtered": ConformerMDSampler._pair_distance_stats(loose_keep_features),
                "refined": ConformerMDSampler._pair_distance_stats(refine_features),
                "final": ConformerMDSampler._pair_distance_stats(selected_features),
            },
            "relax_shift": {
                "pre_to_loose": ConformerMDSampler._rms_displacement_stats(pre_frames, loose_frames),
                "loose_filtered_to_refined": ConformerMDSampler._rms_displacement_stats(loose_keep_frames, refine_frames),
            },
        }

    @staticmethod
    def _safe_ratio(num: int, den: int) -> float | None:
        if den <= 0:
            return None
        return float(num) / float(den)

    @staticmethod
    def _energy_stats(energies: np.ndarray) -> dict:
        e = np.asarray(energies, dtype=float)
        if len(e) == 0:
            return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
        return {
            "n": int(len(e)),
            "min": float(np.min(e)),
            "max": float(np.max(e)),
            "mean": float(np.mean(e)),
            "std": float(np.std(e)),
        }

    @staticmethod
    def _pair_distance_stats(features: np.ndarray) -> dict:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or len(x) < 2:
            return {"n": int(len(x)) if x.ndim == 2 else 0, "pair_min": None, "pair_mean": None, "pair_median": None}
        dists: list[float] = []
        for i in range(len(x)):
            diff = x[i + 1 :] - x[i]
            if len(diff) == 0:
                continue
            d = np.sqrt(np.sum(diff * diff, axis=1))
            dists.extend([float(v) for v in d])
        if not dists:
            return {"n": int(len(x)), "pair_min": None, "pair_mean": None, "pair_median": None}
        arr = np.asarray(dists, dtype=float)
        return {
            "n": int(len(x)),
            "pair_min": float(np.min(arr)),
            "pair_mean": float(np.mean(arr)),
            "pair_median": float(np.median(arr)),
        }

    @staticmethod
    def _rms_displacement_stats(frames_a: list[Atoms], frames_b: list[Atoms]) -> dict:
        if len(frames_a) == 0 or len(frames_b) == 0 or len(frames_a) != len(frames_b):
            return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
        vals: list[float] = []
        for a, b in zip(frames_a, frames_b):
            if len(a) != len(b):
                continue
            da = np.asarray(a.get_positions(), dtype=float)
            db = np.asarray(b.get_positions(), dtype=float)
            rms = float(np.sqrt(np.mean(np.sum((db - da) ** 2, axis=1))))
            vals.append(rms)
        if not vals:
            return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(len(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
        }

    @staticmethod
    def _write_stage_metrics_csv(run_dir: Path, stage_metrics: dict) -> None:
        counts = stage_metrics.get("counts", {})
        retention = stage_metrics.get("retention", {})
        dedup_removed = stage_metrics.get("dedup_removed", {})
        energy = stage_metrics.get("energy", {})
        diversity = stage_metrics.get("diversity", {})
        rows = [
            "metric,value",
            f"count_raw,{counts.get('raw')}",
            f"count_preselected,{counts.get('preselected')}",
            f"count_loose_relaxed,{counts.get('loose_relaxed')}",
            f"count_loose_filtered,{counts.get('loose_filtered')}",
            f"count_refined,{counts.get('refined')}",
            f"count_final,{counts.get('final')}",
            f"retention_pre_over_raw,{retention.get('pre_over_raw')}",
            f"retention_loose_filtered_over_loose,{retention.get('loose_filtered_over_loose')}",
            f"retention_final_over_refine,{retention.get('final_over_refine')}",
            f"retention_final_over_raw,{retention.get('final_over_raw')}",
            f"dedup_loose_filter_removed,{dedup_removed.get('loose_filter_removed')}",
            f"dedup_final_filter_removed,{dedup_removed.get('final_filter_removed')}",
            f"energy_loose_mean,{(energy.get('loose_relaxed') or {}).get('mean')}",
            f"energy_refined_mean,{(energy.get('refined') or {}).get('mean')}",
            f"energy_final_mean,{(energy.get('final') or {}).get('mean')}",
            f"diversity_pre_pair_mean,{(diversity.get('preselected') or {}).get('pair_mean')}",
            f"diversity_refined_pair_mean,{(diversity.get('refined') or {}).get('pair_mean')}",
            f"diversity_final_pair_mean,{(diversity.get('final') or {}).get('pair_mean')}",
        ]
        (run_dir / "stage_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
