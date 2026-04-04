from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import os

import numpy as np
from ase import Atoms
from ase.optimize import BFGS


@dataclass
class MaceRelaxConfig:
    model: str = "small"
    model_path: str | None = None
    device: str = "cuda"
    dtype: str = "float32"
    max_edges_per_batch: int = 15000
    head_name: str | None = None
    enable_cueq: bool = False
    strict: bool = False


def normalize_mace_descriptor_config(model_path: str | None, device: str, dtype: str, strict: bool) -> tuple[str | None, str, str]:
    model_path_env = str(os.environ.get("AE_MACE_MODEL_PATH", "")).strip()
    model_path_use = str(model_path).strip() if model_path is not None else model_path_env
    device_use = str(device).strip() if str(device).strip() else "cpu"
    dtype_map = {"fp32": "float32", "float": "float32", "single": "float32", "fp64": "float64", "double": "float64"}
    dtype_use = dtype_map.get(str(dtype).strip().lower(), str(dtype).strip() if str(dtype).strip() else "float32")
    if device_use.lower().startswith("cuda"):
        try:
            import torch
        except Exception as exc:
            if bool(strict):
                raise RuntimeError("mace_strict=True requires PyTorch with CUDA support.") from exc
            device_use = "cpu"
        else:
            if not bool(torch.cuda.is_available()):
                if bool(strict):
                    raise RuntimeError("mace_strict=True requires torch.cuda.is_available() to be True.")
                device_use = "cpu"
    if bool(strict):
        if not model_path_use or not Path(model_path_use).exists():
            raise FileNotFoundError("mace_strict=True requires an existing model_path (or AE_MACE_MODEL_PATH).")
        if not device_use.lower().startswith("cuda"):
            raise ValueError("mace_strict=True requires device to be cuda.")
    return (model_path_use if model_path_use else None), device_use, dtype_use


def normalize_mace_relax_config(model_path: str | None, device: str, dtype: str, strict: bool) -> tuple[str | None, str, str]:
    model_path_use, device_use, dtype_use = normalize_mace_descriptor_config(
        model_path=model_path, device=device, dtype=dtype, strict=bool(strict)
    )
    if device_use.lower().startswith("cuda") and str(dtype_use).lower() == "float64":
        dtype_use = "float32"
    if bool(strict) and str(dtype_use).lower() != "float32":
        raise ValueError("mace_strict=True requires dtype to be float32 for relax.")
    return model_path_use, device_use, dtype_use


def get_mace_calc(
    model: str = "small",
    model_path: str | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    enable_cueq: bool = False,
    strict: bool = False,
):
    try:
        from mace.calculators import MACECalculator, mace_mp
    except Exception:
        if bool(strict):
            raise
        return None
    if not hasattr(get_mace_calc, "_cache"):
        get_mace_calc._cache = {}
    model_path_use = str(model_path).strip() if model_path is not None else ""
    key = f"{model}|{model_path_use}|{device}|{dtype}|{int(bool(enable_cueq))}|{int(bool(strict))}"
    if key not in get_mace_calc._cache:
        try:
            cueq_use = bool(enable_cueq and str(device).lower().startswith("cuda"))
            if model_path_use and Path(model_path_use).exists():
                get_mace_calc._cache[key] = MACECalculator(
                    model_paths=[model_path_use],
                    device=device,
                    default_dtype=dtype,
                    enable_cueq=cueq_use,
                )
            else:
                if bool(strict):
                    raise FileNotFoundError("mace_strict=True requires an existing model_path (or AE_MACE_MODEL_PATH).")
                get_mace_calc._cache[key] = mace_mp(
                    model=model,
                    device=device,
                    default_dtype=dtype,
                    enable_cueq=cueq_use,
                )
        except Exception:
            if bool(strict):
                raise
            if str(device).lower().startswith("cuda"):
                try:
                    if model_path_use and Path(model_path_use).exists():
                        get_mace_calc._cache[key] = MACECalculator(
                            model_paths=[model_path_use],
                            device="cpu",
                            default_dtype=dtype,
                            enable_cueq=False,
                        )
                    else:
                        get_mace_calc._cache[key] = mace_mp(
                            model=model,
                            device="cpu",
                            default_dtype=dtype,
                            enable_cueq=False,
                        )
                except Exception:
                    return None
            else:
                return None
    return get_mace_calc._cache[key]


class IdentityRelaxBackend:
    def relax(self, frames: list[Atoms], maxf: float, steps: int, work_dir: Path | None = None) -> tuple[list[Atoms], np.ndarray, str]:
        out = [a.copy() for a in frames]
        e = np.zeros(len(out), dtype=float)
        return out, e, "identity"


class MACERelaxBackend:
    def __init__(self, cfg: MaceRelaxConfig):
        self.cfg = cfg

    def relax(self, frames: list[Atoms], maxf: float, steps: int, work_dir: Path | None = None) -> tuple[list[Atoms], np.ndarray, str]:
        if str(os.environ.get("AE_DISABLE_MACE", "")).strip():
            out = [a.copy() for a in frames]
            e = np.zeros(len(out), dtype=float)
            return out, e, "identity_fallback"
        model_path_use, device_use, dtype_use = normalize_mace_relax_config(
            model_path=self.cfg.model_path, device=self.cfg.device, dtype=self.cfg.dtype, strict=bool(self.cfg.strict)
        )
        calc = get_mace_calc(
            model=str(self.cfg.model),
            model_path=model_path_use,
            device=device_use,
            dtype=dtype_use,
            enable_cueq=bool(self.cfg.enable_cueq),
            strict=bool(self.cfg.strict),
        )
        if calc is None:
            if bool(self.cfg.strict):
                raise RuntimeError("MACE is unavailable but mace_strict=True.")
            out = [a.copy() for a in frames]
            e = np.zeros(len(out), dtype=float)
            return out, e, "identity_fallback"
        out_frames: list[Atoms] = []
        energies: list[float] = []
        for f in frames:
            a = f.copy()
            try:
                a.calc = calc
                dyn = BFGS(a, logfile=None)
                dyn.run(fmax=float(maxf), steps=int(steps))
                e = float(a.get_potential_energy())
            except Exception:
                try:
                    e = float(a.get_potential_energy())
                except Exception:
                    e = float("nan")
            out_frames.append(a)
            energies.append(e)
        try:
            used_device = str(getattr(calc, "device", device_use))
        except Exception:
            used_device = str(device_use)
        backend = "mace_calc_file_relax" if model_path_use is not None and Path(str(model_path_use)).exists() else "mace_mp_relax"
        return out_frames, np.asarray(energies, dtype=float), f"{backend}|{used_device}|{dtype_use}|cueq={int(bool(self.cfg.enable_cueq and str(used_device).lower().startswith('cuda')))}"


class MACEBatchRelaxBackend:
    def __init__(self, cfg: MaceRelaxConfig):
        self.cfg = cfg

    def relax(self, frames: list[Atoms], maxf: float, steps: int, work_dir: Path | None = None) -> tuple[list[Atoms], np.ndarray, str]:
        if str(os.environ.get("AE_DISABLE_MACE", "")).strip():
            out = [a.copy() for a in frames]
            e = np.zeros(len(out), dtype=float)
            return out, e, "identity_fallback"
        model_path_use, device_use, dtype_use = normalize_mace_relax_config(
            model_path=self.cfg.model_path, device=self.cfg.device, dtype=self.cfg.dtype, strict=bool(self.cfg.strict)
        )
        calc = get_mace_calc(
            model=str(self.cfg.model),
            model_path=model_path_use,
            device=device_use,
            dtype=dtype_use,
            enable_cueq=bool(self.cfg.enable_cueq),
            strict=bool(self.cfg.strict),
        )
        if calc is None:
            if bool(self.cfg.strict):
                raise RuntimeError("MACE is unavailable but mace_strict=True.")
            out = [a.copy() for a in frames]
            e = np.zeros(len(out), dtype=float)
            return out, e, "identity_fallback"
        from adsorption_ensemble.conformer_md.mace_batch_relax import BatchRelaxer

        if work_dir is None:
            work_dir = Path(".")
        work_dir.mkdir(parents=True, exist_ok=True)
        old = os.environ.get("MACE_BATCHRELAX_DISABLE_TQDM")
        os.environ["MACE_BATCHRELAX_DISABLE_TQDM"] = "1"
        try:
            capture_path = work_dir / "batch_relax.stdout_stderr.log"
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            relaxer = BatchRelaxer(
                calculator=calc,
                max_edges_per_batch=int(self.cfg.max_edges_per_batch),
                device=device_use,
            )
            with capture_path.open("a", encoding="utf-8") as capture_stream:
                with contextlib.redirect_stdout(capture_stream), contextlib.redirect_stderr(capture_stream):
                    relaxed_raw = relaxer.relax(
                        atoms_list=[a.copy() for a in frames],
                        fmax=float(maxf),
                        head=(
                            None
                            if self.cfg.head_name is None or str(self.cfg.head_name).strip() in {"", "Default"}
                            else str(self.cfg.head_name)
                        ),
                        max_n_steps=int(steps),
                        inplace=True,
                        trajectory_dir=(work_dir / "traj").as_posix(),
                        append_trajectory_file=(work_dir / "relaxed_stream.extxyz").as_posix(),
                        save_log_file=(work_dir / "batch_relax.log").as_posix(),
                    )
        finally:
            if old is None:
                os.environ.pop("MACE_BATCHRELAX_DISABLE_TQDM", None)
            else:
                os.environ["MACE_BATCHRELAX_DISABLE_TQDM"] = old
        out_frames: list[Atoms] = []
        energies: list[float] = []
        for i, relaxed in enumerate(relaxed_raw):
            if relaxed is None:
                out_frames.append(frames[i].copy())
                energies.append(float("nan"))
            else:
                out_frames.append(relaxed)
                try:
                    energies.append(float(relaxed.get_potential_energy()))
                except Exception:
                    energies.append(float("nan"))
        try:
            used_device = str(getattr(calc, "device", device_use))
        except Exception:
            used_device = str(device_use)
        backend = (
            "mace_calc_file_batch_relax"
            if model_path_use is not None and Path(str(model_path_use)).exists()
            else "mace_mp_batch_relax"
        )
        return out_frames, np.asarray(energies, dtype=float), f"{backend}|{used_device}|{dtype_use}|cueq={int(bool(self.cfg.enable_cueq and str(used_device).lower().startswith('cuda')))}"
