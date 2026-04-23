from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import gcd
from pathlib import Path
import json
import os
import re

import numpy as np
from ase import Atom, Atoms
from ase.build import bcc100, bcc110, bcc111, bulk, fcc100, fcc110, fcc111, fcc211, molecule, surface
from ase.collections import g2
from ase.io import read, write
from ase.optimize import BFGS

from adsorption_ensemble.pose.sampler import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.basin import BasinBuilder, BasinConfig
from adsorption_ensemble.node import NodeConfig, basin_to_node
from adsorption_ensemble.relax import IdentityRelaxBackend as IdentityRelaxBackendNew, MACEBatchRelaxBackend, MACERelaxBackend, MaceRelaxConfig
from adsorption_ensemble.selection.strategies import DualThresholdSelector, EnergyWindowFilter, FarthestPointSamplingSelector, RMSDSelector
from adsorption_ensemble.conformer_md.descriptors import GeometryPairDistanceDescriptor
from adsorption_ensemble.site import PrimitiveBuilder, PrimitiveEmbedder, PrimitiveEmbeddingConfig, build_site_dictionary
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector, export_surface_detection_report
from adsorption_ensemble.visualization import (
    plot_adsorption_energy_hist,
    plot_deltae_vs_mindist,
    plot_energy_delta_hist,
    plot_feature_pca_compare,
    plot_inequivalent_sites_2d,
    plot_mindist_hist,
    plot_site_centers_only,
    plot_site_embedding_pca,
    plot_surface_primitives_2d,
)

from time import perf_counter


@dataclass
class PoseSweepConfig:
    grid_step: float = 0.6
    spacing: float = 0.8
    l2_distance_threshold: float = 0.22
    max_basis_sites: int | None = None
    n_rotations: int = 8
    n_azimuth: int = 12
    n_shifts: int = 3
    shift_radius: float = 0.25
    n_height_shifts: int = 1
    height_shift_step: float = 0.0
    min_height: float = 1.0
    max_height: float = 3.4
    height_step: float = 0.08
    height_taus: tuple[float, ...] = (0.90, 0.95, 1.00)
    site_contact_tolerance: float = 0.20
    clash_tau: float = 0.85
    max_poses_per_site: int = 24
    random_seed: int = 0
    max_poses_output: int = 256
    postprocess_enabled: bool = True
    postprocess_preselect_k: int = 64
    postprocess_loose_maxf: float = 0.25
    postprocess_loose_steps: int = 20
    postprocess_refine_maxf: float = 0.08
    postprocess_refine_steps: int = 60
    postprocess_energy_window_ev: float = 3.0
    postprocess_rmsd_threshold: float = 0.08
    postprocess_final_energy_window_ev: float = 3.0
    postprocess_final_rmsd_threshold: float = 0.05
    mace_model_path: str = ""
    mace_desc_device: str = "cuda"
    mace_desc_dtype: str = "float64"
    mace_relax_device: str = "cuda"
    mace_relax_dtype: str = "float32"
    mace_max_edges_per_batch: int = 15000
    mace_head_name: str = ""
    mace_strict: bool = False
    postprocess_batch_relax_enabled: bool = True
    profiling_enabled: bool = False
    neighborlist_enabled: bool = True
    neighborlist_min_surface_atoms: int = 64
    neighborlist_cutoff_padding: float = 0.30
    ensemble_enabled: bool = False
    ensemble_relax_backend: str = "mace_batch"
    ensemble_relax_maxf: float = 0.10
    ensemble_relax_steps: int = 80
    ensemble_energy_window_ev: float = 3.0
    ensemble_dedup_metric: str = "mace_node_l2"
    ensemble_rmsd_threshold: float = 0.10
    ensemble_mace_node_l2_threshold: float = 0.20
    ensemble_binding_tau: float = 1.15
    ensemble_desorption_min_bonds: int = 1
    ensemble_surface_reconstruction_max_disp: float = 0.50
    ensemble_dissociation_allow_bond_change: bool = False
    ensemble_burial_margin: float = 0.30
    node_bond_tau: float = 1.20
    node_hash_len: int = 20
    node_identity_mode: str = "legacy_absolute"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def make_vacancy_defect(slab: Atoms, normal_axis: int = 2) -> Atoms:
    s = slab.copy()
    z = s.get_positions()[:, normal_axis]
    idx = int(np.argmax(z))
    del s[idx]
    return s


def make_adatom_defect(slab: Atoms, normal_axis: int = 2) -> Atoms:
    s = slab.copy()
    z = s.get_positions()[:, normal_axis]
    zmax = float(np.max(z))
    top_ids = np.where(z > zmax - 0.2)[0]
    center = np.mean(s.get_positions()[top_ids], axis=0)
    pos = center.copy()
    pos[normal_axis] += 1.8
    top_symbol = s[int(np.argmax(z))].symbol
    s.append(Atom(top_symbol, position=pos))
    return s


def make_alloy_from_top(slab: Atoms, dopant: str, depth: float, every: int, normal_axis: int = 2) -> Atoms:
    s = slab.copy()
    z = s.get_positions()[:, normal_axis]
    zmax = float(np.max(z))
    cand = [int(i) for i in np.where(z > zmax - depth)[0]]
    cand = sorted(cand, key=lambda i: (z[i], i), reverse=True)
    symbols = s.get_chemical_symbols()
    for ii, atom_id in enumerate(cand):
        if ii % every == 0:
            symbols[atom_id] = dopant
    s.set_chemical_symbols(symbols)
    return s


def build_slab_cases() -> dict[str, Atoms]:
    cu_bulk = bulk("Cu", "fcc", a=3.6, cubic=True)
    mgo_bulk = bulk("MgO", "rocksalt", a=4.21, cubic=True)
    base: dict[str, Atoms] = {
        "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
        "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
        "fcc110": fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
        "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
        "bcc100": bcc100("Fe", size=(4, 4, 4), vacuum=12.0),
        "bcc110": bcc110("Fe", size=(4, 4, 4), vacuum=12.0),
        "bcc111": bcc111("Fe", size=(4, 4, 4), vacuum=12.0),
        "fcc111_s3_l5": fcc111("Pt", size=(3, 3, 5), vacuum=12.0),
        "fcc110_s5_l6": fcc110("Pt", size=(5, 4, 6), vacuum=12.0),
        "fcc100_s5_l3": fcc100("Pt", size=(5, 5, 3), vacuum=12.0),
        "fcc211_s9_l5": fcc211("Pt", size=(9, 4, 5), vacuum=12.0),
    }
    hkls = []
    for h in range(1, 5):
        for k in range(0, h + 1):
            for l in range(0, k + 1):
                g = gcd(gcd(h, k), l) if l != 0 else gcd(h, k)
                if g > 1:
                    continue
                hkls.append((h, k, l))
    selected_hkls = [hkl for hkl in hkls if hkl in {(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1), (3, 2, 1)}]
    for hkl in selected_hkls:
        for layers in (3, 4):
            key = f"surface_cu_{hkl[0]}{hkl[1]}{hkl[2]}_l{layers}"
            base[key] = surface(cu_bulk, hkl, layers=layers, vacuum=12.0)
    for hkl in ((1, 0, 0), (1, 1, 1)):
        key = f"mgo_{hkl[0]}{hkl[1]}{hkl[2]}_l4"
        base[key] = surface(mgo_bulk, hkl, layers=4, vacuum=12.0).repeat((2, 2, 1))
    for hkl in ((1, 1, 1), (2, 1, 1)):
        key = f"alloy_cuni_{hkl[0]}{hkl[1]}{hkl[2]}_l4"
        slab = surface(cu_bulk, hkl, layers=4, vacuum=12.0).repeat((2, 2, 1))
        base[key] = make_alloy_from_top(slab, dopant="Ni", depth=2.8, every=3)
    out = dict(base)
    for k in ("fcc111", "fcc110", "fcc211", "mgo_100_l4", "alloy_cuni_111_l4"):
        if k in base:
            out[f"{k}_vacancy"] = make_vacancy_defect(base[k])
            out[f"{k}_adatom"] = make_adatom_defect(base[k])
    return out


def list_supported_molecules(max_count: int | None = None, max_atoms: int = 12) -> list[str]:
    names = sorted(set(g2.names))
    valid = []
    for n in names:
        try:
            mol = molecule(n)
        except Exception:
            continue
        if len(mol) <= max_atoms:
            valid.append(n)
    if max_count is None:
        return valid
    return valid[: max(0, int(max_count))]


def make_surface_atom_features(
    slab: Atoms,
    model: str = "small",
    model_path: str | None = None,
    device: str = "cuda",
    dtype: str = "float64",
    strict: bool = False,
) -> tuple[np.ndarray, str]:
    if str(os.environ.get("AE_DISABLE_MACE", "")).strip():
        z = slab.get_atomic_numbers().astype(float)
        z = z / (np.max(z) + 1e-12)
        return z.reshape(-1, 1), "atomic_number_fallback"
    model_path_use, device_use, dtype_use = _normalize_mace_descriptor_config(
        model_path=model_path, device=device, dtype=dtype, strict=bool(strict)
    )
    calc = _get_mace_mp_calc(model=model, model_path=model_path_use, device=device_use, dtype=dtype_use, strict=bool(strict))
    if calc is None:
        if bool(strict):
            raise RuntimeError("MACE is unavailable but mace_strict=True.")
        z = slab.get_atomic_numbers().astype(float)
        z = z / (np.max(z) + 1e-12)
        return z.reshape(-1, 1), "atomic_number_fallback"
    feats = calc.get_descriptors(slab)
    try:
        used_device = str(getattr(calc, "device", device_use))
    except Exception:
        used_device = str(device_use)
    backend = "mace_calc_file" if model_path_use and Path(model_path_use).exists() else "mace_mp"
    return np.asarray(feats, dtype=float), f"{backend}|{used_device}|{dtype_use}"


def _normalize_mace_descriptor_config(model_path: str | None, device: str, dtype: str, strict: bool) -> tuple[str | None, str, str]:
    model_path_env = str(os.environ.get("AE_MACE_MODEL_PATH", "")).strip()
    model_path_use = str(model_path).strip() if model_path is not None else model_path_env
    device_use = str(device).strip() if str(device).strip() else "cpu"
    dtype_map = {"fp32": "float32", "float": "float32", "single": "float32", "fp64": "float64", "double": "float64"}
    dtype_use = dtype_map.get(str(dtype).strip().lower(), str(dtype).strip() if str(dtype).strip() else "float32")
    if bool(strict):
        if not model_path_use or not Path(model_path_use).exists():
            raise FileNotFoundError("mace_strict=True requires an existing mace_model_path (or AE_MACE_MODEL_PATH).")
        if not device_use.lower().startswith("cuda"):
            raise ValueError("mace_strict=True requires device to be cuda.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("mace_strict=True requires PyTorch with CUDA support.") from exc
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("mace_strict=True requires torch.cuda.is_available() to be True.")
    return (model_path_use if model_path_use else None), device_use, dtype_use


def _normalize_mace_relax_config(model_path: str | None, device: str, dtype: str, strict: bool) -> tuple[str | None, str, str]:
    model_path_use, device_use, dtype_use = _normalize_mace_descriptor_config(
        model_path=model_path, device=device, dtype=dtype, strict=bool(strict)
    )
    if device_use.lower().startswith("cuda") and str(dtype_use).lower() == "float64":
        dtype_use = "float32"
    if bool(strict) and str(dtype_use).lower() != "float32":
        raise ValueError("mace_strict=True requires mace_relax_dtype to be float32.")
    return model_path_use, device_use, dtype_use


def _get_mace_mp_calc(
    model: str = "small",
    model_path: str | None = None,
    device: str = "cuda",
    dtype: str = "float64",
    strict: bool = False,
):
    try:
        from mace.calculators import MACECalculator, mace_mp
    except Exception:
        if bool(strict):
            raise
        return None
    if not hasattr(_get_mace_mp_calc, "_cache"):
        _get_mace_mp_calc._cache = {}
    model_path_use = str(model_path).strip() if model_path is not None else ""
    key = f"{model}|{model_path_use}|{device}|{dtype}|{int(bool(strict))}"
    if key not in _get_mace_mp_calc._cache:
        try:
            if model_path_use and Path(model_path_use).exists():
                _get_mace_mp_calc._cache[key] = MACECalculator(model_paths=[model_path_use], device=device, default_dtype=dtype)
            else:
                if bool(strict):
                    raise FileNotFoundError("mace_strict=True requires an existing mace_model_path (or AE_MACE_MODEL_PATH).")
                _get_mace_mp_calc._cache[key] = mace_mp(model=model, device=device, default_dtype=dtype)
        except Exception:
            if bool(strict):
                raise
            if str(device).lower().startswith("cuda"):
                try:
                    if model_path_use and Path(model_path_use).exists():
                        _get_mace_mp_calc._cache[key] = MACECalculator(
                            model_paths=[model_path_use], device="cpu", default_dtype=dtype
                        )
                    else:
                        _get_mace_mp_calc._cache[key] = mace_mp(model=model, device="cpu", default_dtype=dtype)
                except Exception:
                    return None
            else:
                return None
    return _get_mace_mp_calc._cache[key]


def _pose_mace_features(
    frames: list[Atoms],
    slab_n: int,
    model: str = "small",
    model_path: str | None = None,
    device: str = "cuda",
    dtype: str = "float64",
    strict: bool = False,
) -> tuple[np.ndarray, str]:
    model_path_use, device_use, dtype_use = _normalize_mace_descriptor_config(
        model_path=model_path, device=device, dtype=dtype, strict=bool(strict)
    )
    calc = _get_mace_mp_calc(model=model, model_path=model_path_use, device=device_use, dtype=dtype_use, strict=bool(strict))
    if calc is None:
        if bool(strict):
            raise RuntimeError("MACE is unavailable but mace_strict=True.")
        gdesc = GeometryPairDistanceDescriptor(use_float64=True)
        return np.asarray(gdesc.transform(frames), dtype=float), "geometry_fallback"
    feats = []
    for f in frames:
        desc = np.asarray(calc.get_descriptors(f), dtype=float)
        ads = desc[slab_n:] if len(desc) > slab_n else desc
        vec = np.mean(ads, axis=0)
        feats.append(vec)
    try:
        used_device = str(getattr(calc, "device", device_use))
    except Exception:
        used_device = str(device_use)
    backend = (
        "mace_calc_file_ads_mean" if model_path_use is not None and Path(str(model_path_use)).exists() else "mace_mp_ads_mean"
    )
    return np.asarray(feats, dtype=float), f"{backend}|{used_device}|{dtype_use}"


def _relax_frames_mace_mp(
    frames: list[Atoms],
    maxf: float,
    steps: int,
    model: str = "small",
    model_path: str | None = None,
    device: str = "cuda",
    dtype: str = "float64",
    strict: bool = False,
) -> tuple[list[Atoms], np.ndarray, str]:
    model_path_use, device_use, dtype_use = _normalize_mace_relax_config(
        model_path=model_path, device=device, dtype=dtype, strict=bool(strict)
    )
    calc = _get_mace_mp_calc(model=model, model_path=model_path_use, device=device_use, dtype=dtype_use, strict=bool(strict))
    out_frames: list[Atoms] = []
    energies = []
    if calc is None:
        if bool(strict):
            raise RuntimeError("MACE is unavailable but mace_strict=True.")
        for f in frames:
            out_frames.append(f.copy())
            energies.append(0.0)
        return out_frames, np.asarray(energies, dtype=float), "identity_fallback"
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
                e = np.nan
        out_frames.append(a)
        energies.append(e)
    try:
        used_device = str(getattr(calc, "device", device_use))
    except Exception:
        used_device = str(device_use)
    backend = (
        "mace_calc_file_relax" if model_path_use is not None and Path(str(model_path_use)).exists() else "mace_mp_relax"
    )
    return out_frames, np.asarray(energies, dtype=float), f"{backend}|{used_device}|{dtype_use}"


def _relax_frames_mace_batch(
    frames: list[Atoms],
    maxf: float,
    steps: int,
    model: str = "small",
    model_path: str | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    max_edges_per_batch: int = 15000,
    head_name: str = "Default",
    strict: bool = False,
    work_dir: Path | None = None,
) -> tuple[list[Atoms], np.ndarray, str]:
    model_path_use, device_use, dtype_use = _normalize_mace_relax_config(
        model_path=model_path, device=device, dtype=dtype, strict=bool(strict)
    )
    calc = _get_mace_mp_calc(model=model, model_path=model_path_use, device=device_use, dtype=dtype_use, strict=bool(strict))
    if calc is None:
        if bool(strict):
            raise RuntimeError("MACE is unavailable but mace_strict=True.")
        out_frames = [f.copy() for f in frames]
        energies = np.zeros(len(out_frames), dtype=float)
        return out_frames, energies, "identity_fallback"
    from adsorption_ensemble.conformer_md.mace_batch_relax import BatchRelaxer

    if work_dir is None:
        work_dir = Path(".")
    work_dir.mkdir(parents=True, exist_ok=True)
    old = os.environ.get("MACE_BATCHRELAX_DISABLE_TQDM")
    os.environ["MACE_BATCHRELAX_DISABLE_TQDM"] = "1"
    try:
        relaxer = BatchRelaxer(
            calculator=calc,
            max_edges_per_batch=int(max_edges_per_batch),
            device=device_use,
        )
        head_use = None
        if str(head_name).strip() and str(head_name).strip() != "Default":
            head_use = str(head_name).strip()
        relaxed_raw = relaxer.relax(
            atoms_list=[a.copy() for a in frames],
            fmax=float(maxf),
            head=head_use,
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
    return out_frames, np.asarray(energies, dtype=float), f"{backend}|{used_device}|{dtype_use}"


def _dual_filter_ids(energies: np.ndarray, features: np.ndarray, energy_window_ev: float, rmsd_threshold: float) -> list[int]:
    selector = DualThresholdSelector(
        energy_window=EnergyWindowFilter(delta_e=float(energy_window_ev)),
        rmsd_selector=RMSDSelector(threshold=float(rmsd_threshold)),
    )
    return selector.select(energies=np.asarray(energies, dtype=float), features=np.asarray(features, dtype=float))


def _dual_filter_with_diagnostics(
    case_out: Path,
    stage_name: str,
    energies: np.ndarray,
    features: np.ndarray,
    energy_window_ev: float,
    rmsd_threshold: float,
) -> tuple[list[int], dict]:
    e = np.asarray(energies, dtype=float)
    f = np.asarray(features, dtype=float)
    n_total = int(len(e))
    if n_total == 0 or f.size == 0:
        diag = {
            "stage": str(stage_name),
            "n_total": n_total,
            "energy_window_ev": float(energy_window_ev),
            "rmsd_threshold": float(rmsd_threshold),
            "e_min_ev": None,
            "energy_cut_ev": None,
            "n_energy_keep": 0,
            "n_energy_drop": n_total,
            "n_rmsd_keep": 0,
            "n_rmsd_drop": 0,
        }
        (case_out / f"{stage_name}_filter_diagnostics.json").write_text(
            json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return [], diag
    cand_ids = list(range(n_total))
    e_finite = e[np.isfinite(e)]
    e_min = float(np.min(e_finite)) if e_finite.size else float("nan")
    energy_filter = EnergyWindowFilter(delta_e=float(energy_window_ev))
    energy_keep_ids = energy_filter.select(energies=e, candidate_ids=cand_ids)
    energy_keep_mask = np.zeros(n_total, dtype=bool)
    energy_keep_mask[np.asarray(energy_keep_ids, dtype=int)] = True
    ordered = sorted(energy_keep_ids, key=lambda i: float(e[i]))
    selected: list[int] = []
    rmsd_keep_mask = np.zeros(n_total, dtype=bool)
    mindist = np.full(n_total, np.nan, dtype=float)
    for idx in ordered:
        if not selected:
            selected.append(int(idx))
            rmsd_keep_mask[int(idx)] = True
            mindist[int(idx)] = float("inf")
            continue
        ref = f[np.asarray(selected, dtype=int)]
        diff = ref - f[int(idx)]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        md = float(np.min(dists)) if dists.size else float("nan")
        mindist[int(idx)] = md
        if np.isfinite(md) and md >= float(rmsd_threshold):
            selected.append(int(idx))
            rmsd_keep_mask[int(idx)] = True
    n_energy_keep = int(np.sum(energy_keep_mask))
    n_energy_drop = int(n_total - n_energy_keep)
    n_rmsd_keep = int(np.sum(rmsd_keep_mask))
    n_rmsd_drop = int(n_energy_keep - n_rmsd_keep)
    energy_cut = float(e_min + float(energy_window_ev)) if np.isfinite(e_min) else float("nan")
    diag = {
        "stage": str(stage_name),
        "n_total": n_total,
        "energy_window_ev": float(energy_window_ev),
        "rmsd_threshold": float(rmsd_threshold),
        "e_min_ev": None if not np.isfinite(e_min) else float(e_min),
        "energy_cut_ev": None if not np.isfinite(energy_cut) else float(energy_cut),
        "n_energy_keep": n_energy_keep,
        "n_energy_drop": n_energy_drop,
        "n_rmsd_keep": n_rmsd_keep,
        "n_rmsd_drop": n_rmsd_drop,
        "frac_energy_keep": 0.0 if n_total == 0 else float(n_energy_keep) / float(n_total),
        "frac_energy_drop": 0.0 if n_total == 0 else float(n_energy_drop) / float(n_total),
        "frac_rmsd_keep_of_energy_keep": 0.0 if n_energy_keep == 0 else float(n_rmsd_keep) / float(n_energy_keep),
        "frac_rmsd_drop_of_energy_keep": 0.0 if n_energy_keep == 0 else float(n_rmsd_drop) / float(n_energy_keep),
    }
    (case_out / f"{stage_name}_filter_diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "id,energy_ev,delta_e_ev,energy_keep,min_dist,rmsd_keep",
    ]
    for i in range(n_total):
        ei = e[i]
        de = float(ei - e_min) if np.isfinite(ei) and np.isfinite(e_min) else float("nan")
        md = mindist[i]
        lines.append(
            f"{i},{_format_float(ei)},{_format_float(de)},{int(bool(energy_keep_mask[i]))},{_format_float(md)},{int(bool(rmsd_keep_mask[i]))}"
        )
    (case_out / f"{stage_name}_filter_diagnostics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if np.isfinite(e_min):
        plot_energy_delta_hist(
            energies_ev=e,
            e_min_ev=e_min,
            delta_e_ev=float(energy_window_ev),
            filename=case_out / f"{stage_name}_deltae_hist.png",
            title=f"{stage_name}: ΔE histogram (all candidates)",
        )
    plot_mindist_hist(
        mindist=mindist[energy_keep_mask],
        threshold=float(rmsd_threshold),
        filename=case_out / f"{stage_name}_mindist_hist.png",
        title=f"{stage_name}: min-dist histogram (after ΔE)",
    )
    if np.isfinite(e_min):
        plot_deltae_vs_mindist(
            energies_ev=e,
            e_min_ev=e_min,
            delta_e_ev=float(energy_window_ev),
            mindist=mindist,
            rmsd_threshold=float(rmsd_threshold),
            energy_keep_mask=energy_keep_mask,
            rmsd_keep_mask=rmsd_keep_mask,
            filename=case_out / f"{stage_name}_deltae_vs_mindist.png",
            title=f"{stage_name}: ΔE vs min-dist (filter breakdown)",
        )
    return selected, diag


def _compute_adsorption_energy_artifacts(
    case_out: Path,
    slab_n: int,
    pooled: list[Atoms],
    final_frames: list[Atoms],
    cfg: PoseSweepConfig,
) -> dict:
    if not pooled or slab_n <= 0 or not final_frames:
        return {}
    from adsorption_ensemble.relax.backends import get_mace_calc, normalize_mace_relax_config

    model_path_use, device_use, dtype_use = normalize_mace_relax_config(
        model_path=str(cfg.mace_model_path).strip() if str(cfg.mace_model_path).strip() else None,
        device=str(cfg.mace_relax_device),
        dtype=str(cfg.mace_relax_dtype),
        strict=bool(cfg.mace_strict),
    )
    calc = get_mace_calc(
        model="small",
        model_path=model_path_use,
        device=device_use,
        dtype=dtype_use,
        strict=bool(cfg.mace_strict),
    )
    if calc is None:
        if bool(cfg.mace_strict):
            raise RuntimeError("MACE is unavailable but mace_strict=True.")
        return {}
    ref_comb = pooled[0]
    slab_ref = ref_comb[: int(slab_n)].copy()
    mol_ref = ref_comb[int(slab_n) :].copy()
    if len(mol_ref) <= 0:
        return {}
    try:
        mol_ref.set_cell([20.0, 20.0, 20.0])
        mol_ref.center()
        mol_ref.set_pbc((False, False, False))
    except Exception:
        pass
    slab_ref.calc = calc
    mol_ref.calc = calc
    e_slab = float(slab_ref.get_potential_energy())
    e_mol = float(mol_ref.get_potential_energy())
    e_totals: list[float] = []
    e_ads: list[float] = []
    for a in final_frames:
        e_total = a.info.get("energy_ev", None)
        if e_total is None or not np.isfinite(float(e_total)):
            aa = a.copy()
            aa.calc = calc
            e_total = float(aa.get_potential_energy())
        e_total_f = float(e_total)
        e_totals.append(e_total_f)
        e_ads.append(e_total_f - e_slab - e_mol)
    e_ads_arr = np.asarray(e_ads, dtype=float)
    csv_path = case_out / "adsorption_energy.csv"
    lines = ["id,energy_total_ev,energy_ads_ev"]
    for i in range(len(e_ads)):
        lines.append(f"{i},{_format_float(e_totals[i])},{_format_float(e_ads[i])}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    png_path = case_out / "adsorption_energy_hist.png"
    plot_adsorption_energy_hist(
        adsorption_energies_ev=e_ads_arr,
        filename=png_path,
        title="Final adsorption energy distribution",
    )
    stats = {
        "adsorption_ref_slab_ev": float(e_slab),
        "adsorption_ref_mol_ev": float(e_mol),
        "adsorption_energy_min_ev": None if e_ads_arr.size == 0 else float(np.nanmin(e_ads_arr)),
        "adsorption_energy_mean_ev": None if e_ads_arr.size == 0 else float(np.nanmean(e_ads_arr)),
        "adsorption_energy_median_ev": None if e_ads_arr.size == 0 else float(np.nanmedian(e_ads_arr)),
        "adsorption_energy_csv": csv_path.as_posix(),
        "adsorption_energy_hist_png": png_path.as_posix(),
    }
    return stats


def _run_pose_postprocess(
    case_out: Path,
    slab_n: int,
    pooled: list[Atoms],
    cfg: PoseSweepConfig,
) -> dict:
    if len(pooled) == 0:
        return {
            "enabled": bool(cfg.postprocess_enabled),
            "backend_feature": "none",
            "backend_relax": "none",
            "n_preselected": 0,
            "n_loose": 0,
            "n_loose_keep": 0,
            "n_refine": 0,
            "n_final": 0,
        }
    features, feat_backend = _pose_mace_features(
        pooled,
        slab_n=slab_n,
        model_path=cfg.mace_model_path,
        device=cfg.mace_desc_device,
        dtype=cfg.mace_desc_dtype,
        strict=bool(cfg.mace_strict),
    )
    fps = FarthestPointSamplingSelector(random_seed=cfg.random_seed)
    k = min(len(pooled), max(1, int(cfg.postprocess_preselect_k)))
    pre_ids = fps.select(features=features, k=k)
    plot_feature_pca_compare(
        features=features,
        selected_ids=pre_ids,
        filename=case_out / "pose_preselect_pca.png",
        title="Pose Preselect PCA (All vs Selected)",
        selected_label="Preselected",
        all_label="Pooled",
    )
    pre_frames = [pooled[i].copy() for i in pre_ids]
    write((case_out / "pose_preselected.extxyz").as_posix(), pre_frames)
    if bool(cfg.postprocess_batch_relax_enabled):
        loose_frames, loose_e, relax_backend = _relax_frames_mace_batch(
            pre_frames,
            maxf=cfg.postprocess_loose_maxf,
            steps=cfg.postprocess_loose_steps,
            model_path=cfg.mace_model_path,
            device=cfg.mace_relax_device,
            dtype=cfg.mace_relax_dtype,
            max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
            head_name=str(cfg.mace_head_name),
            strict=bool(cfg.mace_strict),
            work_dir=case_out / "batch_relax_loose",
        )
    else:
        loose_frames, loose_e, relax_backend = _relax_frames_mace_mp(
            pre_frames,
            maxf=cfg.postprocess_loose_maxf,
            steps=cfg.postprocess_loose_steps,
            model_path=cfg.mace_model_path,
            device=cfg.mace_relax_device,
            dtype=cfg.mace_relax_dtype,
            strict=bool(cfg.mace_strict),
        )
    write((case_out / "pose_loose_relaxed.extxyz").as_posix(), loose_frames)
    geom = GeometryPairDistanceDescriptor(use_float64=True)
    loose_feat = np.asarray(geom.transform(loose_frames), dtype=float)
    loose_keep_ids, loose_diag = _dual_filter_with_diagnostics(
        case_out=case_out,
        stage_name="loose",
        energies=np.asarray(loose_e, dtype=float),
        features=np.asarray(loose_feat, dtype=float),
        energy_window_ev=float(cfg.postprocess_energy_window_ev),
        rmsd_threshold=float(cfg.postprocess_rmsd_threshold),
    )
    plot_feature_pca_compare(
        features=loose_feat,
        selected_ids=loose_keep_ids,
        filename=case_out / "pose_loose_filter_pca.png",
        title="Pose Loose Filter PCA (All vs Kept)",
        selected_label="Kept",
        all_label="Loose",
    )
    loose_keep_frames = [loose_frames[i].copy() for i in loose_keep_ids]
    write((case_out / "pose_loose_filtered.extxyz").as_posix(), loose_keep_frames)
    if bool(cfg.postprocess_batch_relax_enabled):
        refine_frames, refine_e, relax_backend_ref = _relax_frames_mace_batch(
            loose_keep_frames,
            maxf=cfg.postprocess_refine_maxf,
            steps=cfg.postprocess_refine_steps,
            model_path=cfg.mace_model_path,
            device=cfg.mace_relax_device,
            dtype=cfg.mace_relax_dtype,
            max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
            head_name=str(cfg.mace_head_name),
            strict=bool(cfg.mace_strict),
            work_dir=case_out / "batch_relax_refine",
        )
    else:
        refine_frames, refine_e, relax_backend_ref = _relax_frames_mace_mp(
            loose_keep_frames,
            maxf=cfg.postprocess_refine_maxf,
            steps=cfg.postprocess_refine_steps,
            model_path=cfg.mace_model_path,
            device=cfg.mace_relax_device,
            dtype=cfg.mace_relax_dtype,
            strict=bool(cfg.mace_strict),
        )
    write((case_out / "pose_refined.extxyz").as_posix(), refine_frames)
    refine_feat = np.asarray(geom.transform(refine_frames), dtype=float) if refine_frames else np.empty((0, 0), dtype=float)
    final_ids, final_diag = (
        _dual_filter_with_diagnostics(
            case_out=case_out,
            stage_name="final",
            energies=np.asarray(refine_e, dtype=float),
            features=np.asarray(refine_feat, dtype=float),
            energy_window_ev=float(cfg.postprocess_final_energy_window_ev),
            rmsd_threshold=float(cfg.postprocess_final_rmsd_threshold),
        )
        if len(refine_frames)
        else ([], {})
    )
    plot_feature_pca_compare(
        features=refine_feat,
        selected_ids=final_ids,
        filename=case_out / "pose_final_filter_pca.png",
        title="Pose Final Filter PCA (All vs Final)",
        selected_label="Final",
        all_label="Refined",
    )
    final_frames = [refine_frames[i].copy() for i in final_ids]
    final_e = np.asarray([refine_e[i] for i in final_ids], dtype=float) if final_ids else np.empty((0,), dtype=float)
    order = np.argsort(np.nan_to_num(final_e, nan=np.inf))
    sorted_frames = [final_frames[i] for i in order]
    sorted_e = [float(final_e[i]) for i in order]
    for i, a in enumerate(sorted_frames):
        a.info["energy_ev"] = sorted_e[i]
    if sorted_frames:
        write((case_out / "pose_final.extxyz").as_posix(), sorted_frames)
    ads_stats = _compute_adsorption_energy_artifacts(
        case_out=case_out,
        slab_n=int(slab_n),
        pooled=pooled,
        final_frames=sorted_frames,
        cfg=cfg,
    )
    metrics = {
        "enabled": True,
        "backend_feature": feat_backend,
        "backend_relax": relax_backend_ref if relax_backend_ref != "identity_fallback" else relax_backend,
        "n_preselected": len(pre_frames),
        "n_loose": len(loose_frames),
        "n_loose_keep": len(loose_keep_frames),
        "n_loose_energy_keep": int(loose_diag.get("n_energy_keep", 0)),
        "n_loose_energy_drop": int(loose_diag.get("n_energy_drop", 0)),
        "n_loose_rmsd_drop": int(loose_diag.get("n_rmsd_drop", 0)),
        "n_refine": len(refine_frames),
        "n_final": len(sorted_frames),
        "n_final_energy_keep": int(final_diag.get("n_energy_keep", 0)) if final_diag else 0,
        "n_final_energy_drop": int(final_diag.get("n_energy_drop", 0)) if final_diag else 0,
        "n_final_rmsd_drop": int(final_diag.get("n_rmsd_drop", 0)) if final_diag else 0,
        "energy_final_min": None if len(sorted_e) == 0 else float(np.nanmin(sorted_e)),
        "energy_final_mean": None if len(sorted_e) == 0 else float(np.nanmean(sorted_e)),
    }
    metrics.update(dict(ads_stats))
    (case_out / "pose_postprocess_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["final_file"] = (case_out / "pose_final.extxyz").as_posix() if sorted_frames else ""
    return metrics


def _select_relax_backend(cfg: PoseSweepConfig):
    if str(cfg.ensemble_relax_backend).strip().lower() in {"none", "identity"}:
        return IdentityRelaxBackendNew(), "identity"
    mace_cfg = MaceRelaxConfig(
        model="small",
        model_path=str(cfg.mace_model_path).strip() if str(cfg.mace_model_path).strip() else None,
        device=str(cfg.mace_relax_device),
        dtype=str(cfg.mace_relax_dtype),
        max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
        head_name=(None if str(cfg.mace_head_name).strip() in {"", "Default"} else str(cfg.mace_head_name).strip()),
        strict=bool(cfg.mace_strict),
    )
    if str(cfg.ensemble_relax_backend).strip().lower() in {"mace", "mace_mp"}:
        return MACERelaxBackend(mace_cfg), "mace"
    if str(cfg.ensemble_relax_backend).strip().lower() in {"mace_batch", "batch"}:
        return MACEBatchRelaxBackend(mace_cfg), "mace_batch"
    return IdentityRelaxBackendNew(), "identity"


def _run_ensemble_generation(
    case_out: Path,
    slab: Atoms,
    mol: Atoms,
    frames: list[Atoms],
    cfg: PoseSweepConfig,
    normal_axis: int,
) -> dict:
    if not frames:
        return {"enabled": True, "n_input": 0, "n_basins": 0, "n_nodes": 0, "basins_file": "", "nodes_file": ""}
    relax_backend, relax_backend_kind = _select_relax_backend(cfg)
    bcfg = BasinConfig(
        relax_maxf=float(cfg.ensemble_relax_maxf),
        relax_steps=int(cfg.ensemble_relax_steps),
        energy_window_ev=float(cfg.ensemble_energy_window_ev),
        dedup_metric=str(cfg.ensemble_dedup_metric),
        rmsd_threshold=float(cfg.ensemble_rmsd_threshold),
        mace_node_l2_threshold=float(cfg.ensemble_mace_node_l2_threshold),
        mace_model_path=str(cfg.mace_model_path).strip() if str(cfg.mace_model_path).strip() else None,
        mace_device=str(cfg.mace_desc_device),
        mace_dtype=str(cfg.mace_desc_dtype),
        mace_max_edges_per_batch=int(cfg.mace_max_edges_per_batch),
        mace_layers_to_keep=-1,
        mace_head_name=(None if str(cfg.mace_head_name).strip() in {"", "Default"} else str(cfg.mace_head_name).strip()),
        mace_mlp_energy_key=None,
        binding_tau=float(cfg.ensemble_binding_tau),
        desorption_min_bonds=int(cfg.ensemble_desorption_min_bonds),
        surface_reconstruction_max_disp=float(cfg.ensemble_surface_reconstruction_max_disp),
        dissociation_allow_bond_change=bool(cfg.ensemble_dissociation_allow_bond_change),
        burial_margin=float(cfg.ensemble_burial_margin),
        work_dir=case_out / "ensemble",
    )
    out = BasinBuilder(config=bcfg, relax_backend=relax_backend).build(
        frames=frames,
        slab_ref=slab,
        adsorbate_ref=mol,
        slab_n=len(slab),
        normal_axis=int(normal_axis),
    )
    basins_file = case_out / "basins.extxyz"
    basins_json = case_out / "basins.json"
    nodes_json = case_out / "nodes.json"
    basins_frames: list[Atoms] = []
    for b in out.basins:
        a = b.atoms.copy()
        a.info["basin_id"] = int(b.basin_id)
        a.info["energy_ev"] = float(b.energy_ev)
        a.info["denticity"] = int(b.denticity)
        a.info["signature"] = str(b.signature)
        a.info["member_candidate_ids"] = ",".join(str(int(x)) for x in b.member_candidate_ids)
        a.info["binding_pairs"] = ",".join(f"{int(i)}:{int(j)}" for i, j in b.binding_pairs)
        basins_frames.append(a)
    if basins_frames:
        write(basins_file.as_posix(), basins_frames)
    basins_payload = {
        "summary": dict(out.summary),
        "relax_backend": str(out.relax_backend),
        "basins": [
            {
                "basin_id": int(b.basin_id),
                "energy_ev": float(b.energy_ev),
                "denticity": int(b.denticity),
                "signature": str(b.signature),
                "member_candidate_ids": [int(x) for x in b.member_candidate_ids],
                "binding_pairs": [(int(i), int(j)) for i, j in b.binding_pairs],
            }
            for b in out.basins
        ],
        "rejected": [
            {"candidate_id": int(r.candidate_id), "reason": str(r.reason), "metrics": dict(r.metrics)} for r in out.rejected
        ],
    }
    basins_json.write_text(json.dumps(basins_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    e0 = out.summary.get("energy_min_ev", None)
    energy_min = None
    if e0 is not None:
        try:
            energy_min = float(e0)
        except Exception:
            energy_min = None
    ncfg = NodeConfig(
        bond_tau=float(cfg.node_bond_tau),
        node_hash_len=int(cfg.node_hash_len),
        node_identity_mode=str(cfg.node_identity_mode),
    )
    nodes = [
        basin_to_node(
            b,
            slab_n=len(slab),
            cfg=ncfg,
            energy_min_ev=energy_min,
            surface_reference=slab,
        )
        for b in out.basins
    ]
    nodes_payload = [
        {
            "node_id": str(n.node_id),
            "node_id_legacy": str(n.node_id_legacy),
            "basin_id": int(n.basin_id),
            "canonical_order": [int(x) for x in n.canonical_order],
            "atomic_numbers": [int(x) for x in n.atomic_numbers],
            "internal_bonds": [(int(i), int(j)) for i, j in n.internal_bonds],
            "binding_pairs": [(int(i), int(j)) for i, j in n.binding_pairs],
            "surface_env_key": (None if n.surface_env_key is None else str(n.surface_env_key)),
            "surface_geometry_key": (None if n.surface_geometry_key is None else str(n.surface_geometry_key)),
            "denticity": int(n.denticity),
            "relative_energy_ev": (None if n.relative_energy_ev is None else float(n.relative_energy_ev)),
            "provenance": dict(n.provenance),
        }
        for n in nodes
    ]
    nodes_json.write_text(json.dumps(nodes_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "enabled": True,
        "backend": str(relax_backend_kind),
        "relax_backend": str(out.relax_backend),
        "n_input": int(out.summary.get("n_input", len(frames))),
        "n_kept": int(out.summary.get("n_kept", 0)),
        "n_basins": int(out.summary.get("n_basins", len(out.basins))),
        "n_nodes": int(len(nodes)),
        "basins_file": basins_file.as_posix() if basins_file.exists() else "",
        "nodes_file": nodes_json.as_posix(),
        "basins_json": basins_json.as_posix(),
    }


def _format_float(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "nan"
    return f"{float(v):.8f}"


def _is_linear_like_molecule(mol: Atoms) -> bool:
    n = len(mol)
    if n <= 1:
        return False
    if n == 2:
        return True
    pos = np.asarray(mol.get_positions(), dtype=float)
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    cov = pos.T @ pos
    evals = np.sort(np.linalg.eigvalsh(cov))
    if evals[-1] <= 1e-12:
        return False
    return bool(evals[1] / evals[-1] < 1e-3)


def _write_pose_summary_csv(path: Path, rows: list[dict]):
    cols = [
        "slab_case",
        "molecule",
        "ok",
        "error",
        "surface_n",
        "primitive_raw_n",
        "primitive_basis_n",
        "primitive_basis_selected_n",
        "basis_compression_ratio",
        "pose_n",
        "pose_height_min",
        "pose_height_mean",
        "pose_height_max",
        "pose_tilt_min_deg",
        "pose_tilt_median_deg",
        "pose_tilt_max_deg",
        "pose_upright_n",
        "upright_coverage_ok",
        "upright_repair_used",
        "height_shift_unique_n",
        "azimuth_unique_n",
        "feature_backend",
        "postprocess_enabled",
        "postprocess_backend",
        "postprocess_final_n",
        "ensemble_enabled",
        "ensemble_backend",
        "ensemble_n_basins",
        "ensemble_n_nodes",
        "ensemble_basins_file",
        "ensemble_nodes_file",
        "selected_detector",
        "output_dir",
        "pool_file",
    ]
    lines = [",".join(cols)]
    for r in rows:
        vals = [str(r.get(c, "")) for c in cols]
        vals = [v.replace(",", ";") for v in vals]
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, run_dir: Path, cfg: PoseSweepConfig, rows: list[dict]):
    lines = [
        "# Pose Sampling Sweep Report",
        "",
        f"- run_dir: {run_dir.as_posix()}",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- grid_step: {cfg.grid_step}",
        f"- spacing: {cfg.spacing}",
        f"- l2_distance_threshold: {cfg.l2_distance_threshold}",
        f"- n_rotations: {cfg.n_rotations}",
        f"- n_azimuth: {cfg.n_azimuth}",
        f"- n_shifts: {cfg.n_shifts}",
        f"- shift_radius: {cfg.shift_radius}",
        f"- n_height_shifts: {cfg.n_height_shifts}",
        f"- height_shift_step: {cfg.height_shift_step}",
        f"- height_taus: {list(cfg.height_taus)}",
        f"- site_contact_tolerance: {cfg.site_contact_tolerance}",
        f"- clash_tau: {cfg.clash_tau}",
        f"- postprocess_enabled: {cfg.postprocess_enabled}",
        f"- ensemble_enabled: {cfg.ensemble_enabled}",
        f"- ensemble_relax_backend: {cfg.ensemble_relax_backend}",
        f"- ensemble_relax_maxf: {cfg.ensemble_relax_maxf}",
        f"- ensemble_relax_steps: {cfg.ensemble_relax_steps}",
        f"- ensemble_energy_window_ev: {cfg.ensemble_energy_window_ev}",
        f"- ensemble_dedup_metric: {cfg.ensemble_dedup_metric}",
        f"- ensemble_rmsd_threshold: {cfg.ensemble_rmsd_threshold}",
        f"- ensemble_mace_node_l2_threshold: {cfg.ensemble_mace_node_l2_threshold}",
        f"- ensemble_binding_tau: {cfg.ensemble_binding_tau}",
        f"- ensemble_desorption_min_bonds: {cfg.ensemble_desorption_min_bonds}",
        f"- node_bond_tau: {cfg.node_bond_tau}",
        f"- node_hash_len: {cfg.node_hash_len}",
        f"- node_identity_mode: {cfg.node_identity_mode}",
        "",
    ]
    ok_rows = [r for r in rows if bool(r.get("ok", False))]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- total_combinations: {len(rows)}")
    lines.append(f"- success_combinations: {len(ok_rows)}")
    lines.append(f"- failed_combinations: {len(rows) - len(ok_rows)}")
    if ok_rows:
        poses = np.array([float(r["pose_n"]) for r in ok_rows], dtype=float)
        comp = np.array([float(r["basis_compression_ratio"]) for r in ok_rows], dtype=float)
        pp_final = np.array([float(r.get("postprocess_final_n", 0)) for r in ok_rows], dtype=float)
        upright = np.array([float(r.get("pose_upright_n", 0)) for r in ok_rows], dtype=float)
        upright_ok = np.array([1.0 if bool(r.get("upright_coverage_ok", False)) else 0.0 for r in ok_rows], dtype=float)
        repaired = np.array([1.0 if bool(r.get("upright_repair_used", False)) else 0.0 for r in ok_rows], dtype=float)
        lines.append(f"- pose_n_min/median/max: {int(np.min(poses))}/{float(np.median(poses)):.1f}/{int(np.max(poses))}")
        lines.append(f"- basis_compression_ratio_min/median/max: {float(np.min(comp)):.4f}/{float(np.median(comp)):.4f}/{float(np.max(comp)):.4f}")
        lines.append(f"- postprocess_final_n_min/median/max: {int(np.min(pp_final))}/{float(np.median(pp_final)):.1f}/{int(np.max(pp_final))}")
        lines.append(f"- pose_upright_n_min/median/max: {int(np.min(upright))}/{float(np.median(upright)):.1f}/{int(np.max(upright))}")
        lines.append(f"- upright_coverage_ok_rate: {float(np.mean(upright_ok)):.3f}")
        lines.append(f"- upright_repair_used_rate: {float(np.mean(repaired)):.3f}")
    lines.append("")
    lines.append("## Top Pose Pools")
    lines.append("")
    lines.append("| slab_case | molecule | pose_n | basis_n(selected/total) | compression | post_final_n | pool_file |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    ranked = sorted(ok_rows, key=lambda x: int(x.get("pose_n", 0)), reverse=True)[:20]
    for r in ranked:
        lines.append(
            f"| {r['slab_case']} | {r['molecule']} | {r['pose_n']} | {r.get('primitive_basis_selected_n',0)}/{r['primitive_basis_n']} | {float(r['basis_compression_ratio']):.4f} | {r.get('postprocess_final_n',0)} | {r['pool_file']} |"
        )
    lines.append("")
    if any(bool(r.get("ensemble_enabled", False)) for r in ok_rows):
        lines.append("## Top Ensembles")
        lines.append("")
        lines.append("| slab_case | molecule | basins_n | nodes_n | basins_file | nodes_file |")
        lines.append("|---|---|---:|---:|---|---|")
        ranked_e = sorted(ok_rows, key=lambda x: int(x.get("ensemble_n_basins", 0)), reverse=True)[:20]
        for r in ranked_e:
            if not bool(r.get("ensemble_enabled", False)):
                continue
            lines.append(
                f"| {r['slab_case']} | {r['molecule']} | {r.get('ensemble_n_basins',0)} | {r.get('ensemble_n_nodes',0)} | {r.get('ensemble_basins_file','')} | {r.get('ensemble_nodes_file','')} |"
            )
        lines.append("")
    lines.append("## Failures")
    lines.append("")
    failed = [r for r in rows if not bool(r.get("ok", False))]
    if not failed:
        lines.append("- none")
    else:
        for r in failed[:100]:
            lines.append(f"- {r['slab_case']} + {r['molecule']}: {r.get('error', '')}")
    lines.append("")
    lines.append("## User Guided Changes")
    lines.append("")
    lines.append("- 用户要求基于不等价位点最小子集做 site-conditioned 采样，不再使用全部位点。")
    lines.append("- 用户要求采用局域坐标系 + MIC + 自适应高度 + 碰撞约束生成初猜结构。")
    lines.append("- 用户要求生成后执行预采样-粗弛豫-双阈值筛选-精弛豫-最终排序；本次输出包含各阶段结构与指标。")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pose_sampling_sweep(
    out_root: Path,
    cfg: PoseSweepConfig,
    max_molecules: int | None = None,
    max_slabs: int | None = None,
    max_atoms_per_molecule: int = 12,
    max_combinations: int | None = None,
    slab_filter: list[str] | None = None,
) -> dict:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    slab_cases = build_slab_cases()
    slab_names = sorted(slab_cases.keys())
    if slab_filter:
        allow = set(str(x) for x in slab_filter)
        slab_names = [n for n in slab_names if n in allow]
    if max_slabs is not None:
        slab_names = slab_names[: max(0, int(max_slabs))]
    mol_names = list_supported_molecules(max_count=max_molecules, max_atoms=max_atoms_per_molecule)
    rows: list[dict] = []
    n_combo = 0
    for slab_name in slab_names:
        slab = slab_cases[slab_name]
        t_slab_pre = 0.0
        t_slab_primitives = 0.0
        t_slab_features = 0.0
        t_slab_embedding = 0.0
        t_slab_viz_io = 0.0
        pre = SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=cfg.grid_step),
            fallback_detector=VoxelFloodDetector(spacing=cfg.spacing),
            target_surface_fraction=0.25,
            target_count_mode="off",
        )
        t0 = perf_counter()
        ctx = pre.build_context(slab)
        t_slab_pre = float(perf_counter() - t0)
        t0 = perf_counter()
        primitives = PrimitiveBuilder(min_site_distance=0.1).build(slab, ctx)
        t_slab_primitives = float(perf_counter() - t0)
        if len(primitives) == 0:
            for mol_name in mol_names:
                rows.append(
                    {
                        "slab_case": slab_name,
                        "molecule": mol_name,
                        "ok": False,
                        "error": "no_primitives",
                        "surface_n": len(ctx.detection.surface_atom_ids),
                    }
                )
            continue
        t0 = perf_counter()
        atom_features, feature_backend = make_surface_atom_features(
            slab,
            model_path=cfg.mace_model_path,
            device=cfg.mace_desc_device,
            dtype=cfg.mace_desc_dtype,
            strict=bool(cfg.mace_strict),
        )
        t_slab_features = float(perf_counter() - t0)
        emb = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=cfg.l2_distance_threshold))
        t0 = perf_counter()
        emb_res = emb.fit_transform(slab=slab, primitives=primitives, atom_features=atom_features)
        t_slab_embedding = float(perf_counter() - t0)
        basis_all = list(emb_res.basis_primitives)
        if cfg.max_basis_sites is None:
            basis = basis_all
        else:
            basis = basis_all[: max(1, int(cfg.max_basis_sites))]
        slab_dir = run_dir / _safe_name(slab_name)
        slab_dir.mkdir(parents=True, exist_ok=True)
        t0 = perf_counter()
        export_surface_detection_report(slab, ctx, slab_dir / "surface_report")
        plot_surface_primitives_2d(slab, ctx, emb_res.primitives, slab_dir / "sites.png")
        plot_site_centers_only(slab, emb_res.primitives, slab_dir / "sites_only.png")
        plot_inequivalent_sites_2d(slab, emb_res.primitives, slab_dir / "sites_inequivalent.png")
        plot_site_embedding_pca(emb_res.primitives, slab_dir / "site_embedding_pca.png")
        site_dict = build_site_dictionary(emb_res.primitives, slab=slab)
        (slab_dir / "site_dictionary.json").write_text(json.dumps(site_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        t_slab_viz_io = float(perf_counter() - t0)
        for mol_name in mol_names:
            if max_combinations is not None and n_combo >= int(max_combinations):
                break
            n_combo += 1
            case_out = slab_dir / _safe_name(mol_name)
            case_out.mkdir(parents=True, exist_ok=True)
            row = {
                "slab_case": slab_name,
                "molecule": mol_name,
                "ok": False,
                "error": "",
                "surface_n": len(ctx.detection.surface_atom_ids),
                "primitive_raw_n": emb_res.raw_count,
                "primitive_basis_n": emb_res.basis_count,
                "primitive_basis_selected_n": len(basis),
                "basis_compression_ratio": emb_res.compression_ratio,
                "pose_n": 0,
                "pose_height_min": np.nan,
                "pose_height_mean": np.nan,
                "pose_height_max": np.nan,
                "pose_tilt_min_deg": np.nan,
                "pose_tilt_median_deg": np.nan,
                "pose_tilt_max_deg": np.nan,
                "pose_upright_n": 0,
                "upright_coverage_ok": False,
                "upright_repair_used": False,
                "height_shift_unique_n": 0,
                "azimuth_unique_n": 0,
                "feature_backend": feature_backend,
                "postprocess_enabled": bool(cfg.postprocess_enabled),
                "postprocess_backend": "",
                "postprocess_final_n": 0,
                "ensemble_enabled": bool(cfg.ensemble_enabled),
                "ensemble_backend": "",
                "ensemble_n_basins": 0,
                "ensemble_n_nodes": 0,
                "ensemble_basins_file": "",
                "ensemble_nodes_file": "",
                "selected_detector": ctx.detection.diagnostics.get("selected_detector", ""),
                "output_dir": case_out.as_posix(),
                "pool_file": "",
            }
            if bool(cfg.profiling_enabled):
                row.update(
                    {
                        "timing_slab_preprocess_s": float(t_slab_pre),
                        "timing_slab_primitives_s": float(t_slab_primitives),
                        "timing_slab_features_s": float(t_slab_features),
                        "timing_slab_embedding_s": float(t_slab_embedding),
                        "timing_slab_viz_io_s": float(t_slab_viz_io),
                        "timing_sampling_s": 0.0,
                        "timing_write_pool_s": 0.0,
                        "timing_postprocess_s": 0.0,
                        "sampler_profile": {},
                        "mace_model_path": str(cfg.mace_model_path),
                        "mace_desc_device": str(cfg.mace_desc_device),
                        "mace_desc_dtype": str(cfg.mace_desc_dtype),
                        "mace_relax_device": str(cfg.mace_relax_device),
                        "mace_relax_dtype": str(cfg.mace_relax_dtype),
                        "mace_max_edges_per_batch": int(cfg.mace_max_edges_per_batch),
                        "mace_head_name": str(cfg.mace_head_name),
                        "postprocess_batch_relax_enabled": bool(cfg.postprocess_batch_relax_enabled),
                    }
                )
            try:
                mol = molecule(mol_name)
                base_sampler_cfg = PoseSamplerConfig(
                    n_rotations=cfg.n_rotations,
                    n_azimuth=cfg.n_azimuth,
                    n_shifts=cfg.n_shifts,
                    shift_radius=cfg.shift_radius,
                    n_height_shifts=cfg.n_height_shifts,
                    height_shift_step=cfg.height_shift_step,
                    min_height=cfg.min_height,
                    max_height=cfg.max_height,
                    height_step=cfg.height_step,
                    height_taus=tuple(float(x) for x in cfg.height_taus),
                    site_contact_tolerance=cfg.site_contact_tolerance,
                    clash_tau=cfg.clash_tau,
                    max_poses_per_site=cfg.max_poses_per_site,
                    random_seed=cfg.random_seed,
                    profiling_enabled=bool(cfg.profiling_enabled),
                    neighborlist_enabled=bool(cfg.neighborlist_enabled),
                    neighborlist_min_surface_atoms=int(cfg.neighborlist_min_surface_atoms),
                    neighborlist_cutoff_padding=float(cfg.neighborlist_cutoff_padding),
                )
                sampler = PoseSampler(base_sampler_cfg)
                t0 = perf_counter()
                poses = sampler.sample(
                    slab=slab,
                    adsorbate=mol,
                    primitives=basis,
                    surface_atom_ids=ctx.detection.surface_atom_ids,
                )
                if bool(cfg.profiling_enabled):
                    row["timing_sampling_s"] = float(perf_counter() - t0)
                    row["sampler_profile"] = dict(getattr(sampler, "last_profile", {}) or {})
                    (case_out / "profiling_pose_sampler.json").write_text(
                        json.dumps(row["sampler_profile"], ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                linear_like = _is_linear_like_molecule(mol)
                repair_used = False
                if linear_like:
                    initial_upright = int(np.sum(np.asarray([p.tilt_deg for p in poses], dtype=float) <= 30.0)) if poses else 0
                    if initial_upright == 0:
                        repair_used = True
                        repair_cfg = PoseSamplerConfig(
                            n_rotations=max(12, cfg.n_rotations * 2),
                            n_azimuth=max(16, cfg.n_azimuth * 2),
                            n_shifts=max(4, cfg.n_shifts),
                            shift_radius=cfg.shift_radius,
                            n_height_shifts=max(3, cfg.n_height_shifts),
                            height_shift_step=max(0.08, cfg.height_shift_step),
                            min_height=cfg.min_height,
                            max_height=cfg.max_height + 0.8,
                            height_step=max(0.05, cfg.height_step * 0.75),
                            height_taus=tuple(sorted(set(tuple(float(x) for x in cfg.height_taus) + (0.80, 0.85, 0.90, 1.00)))),
                            site_contact_tolerance=cfg.site_contact_tolerance + 0.25,
                            clash_tau=cfg.clash_tau,
                            max_poses_per_site=max(cfg.max_poses_per_site, 36),
                            random_seed=cfg.random_seed + 17,
                            profiling_enabled=bool(cfg.profiling_enabled),
                            neighborlist_enabled=bool(cfg.neighborlist_enabled),
                            neighborlist_min_surface_atoms=int(cfg.neighborlist_min_surface_atoms),
                            neighborlist_cutoff_padding=float(cfg.neighborlist_cutoff_padding),
                        )
                        poses_repair = PoseSampler(repair_cfg).sample(
                            slab=slab,
                            adsorbate=mol,
                            primitives=basis,
                            surface_atom_ids=ctx.detection.surface_atom_ids,
                        )
                        poses = poses + poses_repair
                        poses.sort(key=lambda x: x.height)
                        poses = poses[: max(1, int(cfg.max_poses_output))]
                poses = poses[: max(1, int(cfg.max_poses_output))]
                pooled: list[Atoms] = []
                heights = []
                tilts = []
                azimuth_ids = set()
                hshift_ids = set()
                for i, p in enumerate(poses):
                    comb = slab + p.atoms
                    comb.info["slab_case"] = slab_name
                    comb.info["molecule"] = mol_name
                    comb.info["pose_id"] = int(i)
                    comb.info["primitive_index"] = int(p.primitive_index)
                    comb.info["basis_id"] = -1 if p.basis_id is None else int(p.basis_id)
                    comb.info["site_atom_ids"] = ",".join(str(int(x)) for x in basis[p.primitive_index].atom_ids)
                    comb.info["site_normal_x"] = float(basis[p.primitive_index].normal[0])
                    comb.info["site_normal_y"] = float(basis[p.primitive_index].normal[1])
                    comb.info["site_normal_z"] = float(basis[p.primitive_index].normal[2])
                    comb.info["rotation_index"] = int(p.rotation_index)
                    comb.info["azimuth_index"] = int(p.azimuth_index)
                    comb.info["azimuth_rad"] = float(p.azimuth_rad)
                    comb.info["height_shift_index"] = int(p.height_shift_index)
                    comb.info["height_shift_delta"] = float(p.height_shift_delta)
                    comb.info["tilt_deg"] = float(p.tilt_deg)
                    comb.info["height"] = float(p.height)
                    pooled.append(comb)
                    heights.append(float(p.height))
                    tilts.append(float(p.tilt_deg))
                    azimuth_ids.add(int(p.azimuth_index))
                    hshift_ids.add(int(p.height_shift_index))
                pool_file = case_out / "pose_pool.extxyz"
                if pooled:
                    t0 = perf_counter()
                    write(pool_file.as_posix(), pooled)
                    if bool(cfg.profiling_enabled):
                        row["timing_write_pool_s"] = float(perf_counter() - t0)
                    row["pool_file"] = pool_file.as_posix()
                    row["pose_n"] = len(pooled)
                    row["pose_height_min"] = float(np.min(heights))
                    row["pose_height_mean"] = float(np.mean(heights))
                    row["pose_height_max"] = float(np.max(heights))
                    row["pose_tilt_min_deg"] = float(np.min(tilts))
                    row["pose_tilt_median_deg"] = float(np.median(tilts))
                    row["pose_tilt_max_deg"] = float(np.max(tilts))
                    row["pose_upright_n"] = int(np.sum(np.asarray(tilts, dtype=float) <= 30.0))
                    row["upright_coverage_ok"] = bool(row["pose_upright_n"] > 0)
                    row["upright_repair_used"] = bool(repair_used)
                    row["height_shift_unique_n"] = len(hshift_ids)
                    row["azimuth_unique_n"] = len(azimuth_ids)
                    pp: dict = {}
                    if cfg.postprocess_enabled:
                        t1 = perf_counter()
                        pp = _run_pose_postprocess(case_out=case_out, slab_n=len(slab), pooled=pooled, cfg=cfg)
                        if bool(cfg.profiling_enabled):
                            row["timing_postprocess_s"] = float(perf_counter() - t1)
                        row["postprocess_backend"] = pp.get("backend_relax", "")
                        row["postprocess_final_n"] = int(pp.get("n_final", 0))
                    if bool(cfg.ensemble_enabled):
                        if bool(cfg.postprocess_enabled) and str(pp.get("final_file", "")).strip():
                            frames_for_ens = list(read(str(pp["final_file"]), ":"))
                        else:
                            frames_for_ens = pooled
                        ens = _run_ensemble_generation(
                            case_out=case_out,
                            slab=slab,
                            mol=mol,
                            frames=frames_for_ens,
                            cfg=cfg,
                            normal_axis=int(ctx.classification.normal_axis),
                        )
                        row["ensemble_backend"] = str(ens.get("backend", ""))
                        row["ensemble_n_basins"] = int(ens.get("n_basins", 0))
                        row["ensemble_n_nodes"] = int(ens.get("n_nodes", 0))
                        row["ensemble_basins_file"] = str(ens.get("basins_file", ""))
                        row["ensemble_nodes_file"] = str(ens.get("nodes_file", ""))
                row["ok"] = True
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
        if max_combinations is not None and n_combo >= int(max_combinations):
            break
    summary_json = run_dir / "pose_sampling_summary.json"
    summary_csv = run_dir / "pose_sampling_summary.csv"
    report_md = run_dir / "pose_sampling_report.md"
    payload = {"run_dir": run_dir.as_posix(), "rows": rows}
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_pose_summary_csv(summary_csv, rows)
    _write_report(report_md, run_dir, cfg, rows)
    out = {
        "run_dir": run_dir.as_posix(),
        "summary_json": summary_json.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "report_md": report_md.as_posix(),
        "rows": rows,
    }
    return out


def summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    ok = [r for r in rows if bool(r.get("ok", False))]
    fail = [r for r in rows if not bool(r.get("ok", False))]
    pose_vals = np.array([int(r.get("pose_n", 0)) for r in ok], dtype=float) if ok else np.array([], dtype=float)
    compression = (
        np.array([float(r.get("basis_compression_ratio", np.nan)) for r in ok], dtype=float) if ok else np.array([], dtype=float)
    )
    family_counter = Counter()
    for r in ok:
        name = str(r.get("slab_case", ""))
        if name.startswith("alloy_"):
            family_counter["alloy"] += 1
        elif name.startswith("mgo_"):
            family_counter["oxide"] += 1
        elif name.startswith("bcc"):
            family_counter["bcc_metal"] += 1
        elif name.startswith("fcc"):
            family_counter["fcc_metal"] += 1
        else:
            family_counter["other"] += 1
    return {
        "total": total,
        "ok": len(ok),
        "fail": len(fail),
        "pose_n_min": int(np.min(pose_vals)) if len(pose_vals) else 0,
        "pose_n_median": float(np.median(pose_vals)) if len(pose_vals) else 0.0,
        "pose_n_max": int(np.max(pose_vals)) if len(pose_vals) else 0,
        "compression_min": float(np.nanmin(compression)) if len(compression) else np.nan,
        "compression_median": float(np.nanmedian(compression)) if len(compression) else np.nan,
        "compression_max": float(np.nanmax(compression)) if len(compression) else np.nan,
        "family_counts": dict(family_counter),
    }


def summary_to_text(summary: dict) -> str:
    return (
        f"total={summary['total']} ok={summary['ok']} fail={summary['fail']} "
        f"pose_n(min/med/max)={summary['pose_n_min']}/{summary['pose_n_median']:.1f}/{summary['pose_n_max']} "
        f"compression(min/med/max)={_format_float(summary['compression_min'])}/{_format_float(summary['compression_median'])}/{_format_float(summary['compression_max'])}"
    )
