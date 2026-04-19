from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry.geometry import find_mic
from ase.geometry.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.neighborlist import NeighborList
from time import perf_counter
import itertools

from adsorption_ensemble.site.primitives import SitePrimitive

_PAULING_ELECTRONEGATIVITY = {
    1: 2.20,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    35: 2.96,
    53: 2.66,
}

_POSE_ORTHO_MIN_RATIO_NUMBA_SENTINEL = object()
_POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL = _POSE_ORTHO_MIN_RATIO_NUMBA_SENTINEL


def _get_pose_orthogonal_min_ratio_kernel():
    global _POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL
    if _POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL is None:
        return None
    if _POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL is not _POSE_ORTHO_MIN_RATIO_NUMBA_SENTINEL:
        return _POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL
    try:
        import numba
    except Exception:
        _POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL = None
        return None

    @numba.njit(cache=True, fastmath=True)
    def min_ratio2_orthogonal(
        placed_pos: np.ndarray,
        surf_frac: np.ndarray,
        inv_cell: np.ndarray,
        lengths: np.ndarray,
        pbc_mask: np.ndarray,
        denom2: np.ndarray,
        site_ids_local: np.ndarray,
    ) -> tuple[float, float]:
        na = placed_pos.shape[0]
        ns = surf_frac.shape[0]
        min_surface2 = np.inf
        min_site2 = np.inf
        has_site = site_ids_local.shape[0] > 0
        for ai in range(na):
            fx = (
                placed_pos[ai, 0] * inv_cell[0, 0]
                + placed_pos[ai, 1] * inv_cell[1, 0]
                + placed_pos[ai, 2] * inv_cell[2, 0]
            )
            fy = (
                placed_pos[ai, 0] * inv_cell[0, 1]
                + placed_pos[ai, 1] * inv_cell[1, 1]
                + placed_pos[ai, 2] * inv_cell[2, 1]
            )
            fz = (
                placed_pos[ai, 0] * inv_cell[0, 2]
                + placed_pos[ai, 1] * inv_cell[1, 2]
                + placed_pos[ai, 2] * inv_cell[2, 2]
            )
            for sj in range(ns):
                dx = surf_frac[sj, 0] - fx
                dy = surf_frac[sj, 1] - fy
                dz = surf_frac[sj, 2] - fz
                if pbc_mask[0]:
                    dx = dx - np.round(dx)
                if pbc_mask[1]:
                    dy = dy - np.round(dy)
                if pbc_mask[2]:
                    dz = dz - np.round(dz)
                d2 = dx * dx * lengths[0] * lengths[0] + dy * dy * lengths[1] * lengths[1] + dz * dz * lengths[2] * lengths[2]
                ratio2 = d2 / denom2[ai, sj]
                if ratio2 < min_surface2:
                    min_surface2 = ratio2
                if has_site:
                    for kk in range(site_ids_local.shape[0]):
                        if sj == site_ids_local[kk]:
                            if ratio2 < min_site2:
                                min_site2 = ratio2
                            break
        return min_site2, min_surface2

    _POSE_ORTHO_MIN_RATIO_NUMBA_KERNEL = min_ratio2_orthogonal
    return min_ratio2_orthogonal


@dataclass
class PoseSamplerConfig:
    placement_mode: str = "anchor_free"
    anchor_free_reference: str = "center_of_mass"
    n_rotations: int = 12
    n_azimuth: int = 12
    n_shifts: int = 4
    shift_radius: float = 0.35
    n_height_shifts: int = 1
    height_shift_step: float = 0.0
    min_height: float = 1.2
    max_height: float = 3.5
    height_step: float = 0.1
    height_taus: tuple[float, ...] = (0.90, 0.95, 1.00)
    site_contact_tolerance: float = 0.20
    diatomic_contact_extra_tolerance: float = 0.60
    linear_contact_extra_tolerance: float = 1.00
    clash_tau: float = 0.85
    prune_com_distance: float = 0.25
    prune_rot_distance: float = 0.35
    max_poses_per_site: int | None = None
    random_seed: int = 0
    height_bisect_steps: int = 8
    profiling_enabled: bool = False
    neighborlist_enabled: bool = True
    neighborlist_min_surface_atoms: int = 64
    neighborlist_cutoff_padding: float = 0.30
    nonlinear_atom_down_coverage: bool = True
    nonlinear_atom_down_max_vectors: int = 4
    adaptive_height_fallback: bool = True
    adaptive_height_fallback_step: float = 0.20
    adaptive_height_fallback_max_extra: float = 1.60
    adaptive_height_fallback_contact_slack: float = 0.60


@dataclass
class PoseCandidate:
    primitive_index: int
    basis_id: int | None
    rotation_index: int
    azimuth_index: int
    azimuth_rad: float
    height_shift_index: int
    height_shift_delta: float
    tilt_deg: float
    shift_uv: np.ndarray
    quaternion: np.ndarray
    height: float
    com: np.ndarray
    atoms: Atoms


class PoseSampler:
    def __init__(self, config: PoseSamplerConfig | None = None):
        self.config = config or PoseSamplerConfig()
        self.last_profile: dict = {}
        self._prof: dict | None = None
        self._nl: NeighborList | None = None
        self._surface_slab: Atoms | None = None
        self._surface_slab_map: dict[int, int] | None = None
        self._surface_local_ids: list[int] | None = None
        self._surface_r: np.ndarray | None = None
        self._surface_pos: np.ndarray | None = None
        self._surface_frac: np.ndarray | None = None
        self._cell: np.ndarray | None = None
        self._inv_cell: np.ndarray | None = None
        self._pbc: np.ndarray | None = None
        self._cell_orthogonal: bool = False
        self._cell_lengths: np.ndarray | None = None
        self._ads_r: np.ndarray | None = None
        self._denom2_surface: np.ndarray | None = None
        self._mic_rcell: np.ndarray | None = None
        self._mic_inv_rcell: np.ndarray | None = None
        self._mic_vrvecs: np.ndarray | None = None
        self._nl_tau_factor: float = self._resolve_neighborlist_tau_factor()
        self._ortho_min_ratio_kernel = _get_pose_orthogonal_min_ratio_kernel()

    def sample(
        self,
        slab: Atoms,
        adsorbate: Atoms,
        primitives: list[SitePrimitive],
        surface_atom_ids: list[int],
    ) -> list[PoseCandidate]:
        prof: dict | None = None
        if bool(getattr(self.config, "profiling_enabled", False)):
            prof = {
                "n_solve_height": 0,
                "t_solve_height_s": 0.0,
                "n_height_checks": 0,
                "t_height_checks_s": 0.0,
                "n_has_clash": 0,
                "t_has_clash_s": 0.0,
                "n_height_none": 0,
                "n_candidates_raw": 0,
                "t_prune_s": 0.0,
                "n_candidates_before_prune": 0,
                "n_candidates_after_prune": 0,
                "neighborlist_used": False,
                "neighborlist_surface_n": 0,
                "neighborlist_min_surface_atoms": int(self.config.neighborlist_min_surface_atoms),
            }
        self._prof = prof
        if len(adsorbate) == 0:
            raise ValueError("Adsorbate is empty.")
        if len(primitives) == 0:
            return []
        if len(surface_atom_ids) == 0:
            raise ValueError("surface_atom_ids is empty.")
        mol_class = self._classify_molecule_shape(adsorbate)
        n_rot, n_az, n_shift = self._effective_sampling_budget(mol_class)
        rng = np.random.default_rng(self.config.random_seed)
        azimuth_list = self._sample_azimuth_angles(n_az)
        shift_list = self._sample_tangent_shifts(rng, n_shift, self.config.shift_radius)
        height_shift_list = self._sample_height_shifts(int(self.config.n_height_shifts), float(self.config.height_shift_step))
        ads_base = adsorbate.copy()
        ads_base.positions = self._center_adsorbate_for_sampling(adsorbate)
        self._cell = np.asarray(slab.cell.array, dtype=float)
        self._pbc = np.asarray(slab.get_pbc(), dtype=bool)
        try:
            self._inv_cell = np.asarray(np.linalg.inv(self._cell), dtype=float)
        except np.linalg.LinAlgError:
            self._inv_cell = None
        self._cell_orthogonal = bool(self._is_cell_orthogonal(self._cell))
        self._cell_lengths = np.asarray(np.linalg.norm(self._cell, axis=1), dtype=float)
        self._prepare_mic_cache()
        self._surface_slab = slab[surface_atom_ids]
        self._surface_slab_map = {int(sid): int(i) for i, sid in enumerate(surface_atom_ids)}
        self._surface_local_ids = list(range(int(len(self._surface_slab))))
        z = self._surface_slab.get_atomic_numbers()
        self._surface_r = np.asarray([covalent_radii[int(v)] for v in z], dtype=float)
        self._surface_pos = np.asarray(self._surface_slab.positions, dtype=float)
        if self._inv_cell is not None:
            self._surface_frac = np.asarray(self._surface_pos @ self._inv_cell, dtype=float)
        else:
            self._surface_frac = None
        ads_numbers = ads_base.get_atomic_numbers()
        self._ads_r = np.asarray([covalent_radii[int(v)] for v in ads_numbers], dtype=float)
        self._denom2_surface = np.asarray((self._ads_r[:, None] + self._surface_r[None, :]) ** 2, dtype=float)
        self._setup_neighborlist(slab=self._surface_slab, adsorbate=ads_base)
        if prof is not None:
            prof["neighborlist_used"] = bool(self._nl is not None)
            prof["neighborlist_surface_n"] = int(len(self._surface_local_ids))
        candidates: list[PoseCandidate] = []
        for pidx, primitive in enumerate(primitives):
            site_candidates: list[PoseCandidate] = []
            site_ids_local = self._map_site_atom_ids_to_surface(primitive.atom_ids)
            if site_ids_local is None:
                continue
            quat_list = self._build_site_oriented_quaternions(
                adsorbate_centered=ads_base,
                normal=primitive.normal,
                mol_class=mol_class,
                rng=rng,
                n_rot=n_rot,
            )
            for shift_uv in shift_list:
                center = primitive.center + shift_uv[0] * primitive.t1 + shift_uv[1] * primitive.t2
                for ridx, quat in enumerate(quat_list):
                    rotated = self._rotated_adsorbate(ads_base, quat)
                    for aidx, az in enumerate(azimuth_list):
                        rotated_az = self._rotate_around_axis(rotated, primitive.normal, az)
                        q_az = self._rotation_quaternion(primitive.normal, az)
                        q_tot = self._quaternion_multiply(q_az, quat)
                        t0 = perf_counter() if prof is not None else 0.0
                        if prof is not None:
                            prof["n_solve_height"] += 1
                        h = self._solve_height(
                            slab=slab,
                            ads_positions=np.asarray(rotated_az.positions, dtype=float),
                            center=center,
                            normal=primitive.normal,
                            mol_class=mol_class,
                            site_ids_local=site_ids_local,
                        )
                        if prof is not None:
                            prof["t_solve_height_s"] += float(perf_counter() - t0)
                        if h is None:
                            if bool(self.config.adaptive_height_fallback):
                                candidate = self._build_adaptive_height_candidate(
                                    primitive_index=int(pidx),
                                    basis_id=primitive.basis_id,
                                    rotation_index=int(ridx),
                                    azimuth_index=int(aidx),
                                    azimuth_rad=float(az),
                                    shift_uv=np.asarray(shift_uv, dtype=float),
                                    quaternion=np.asarray(q_tot, dtype=float),
                                    center=np.asarray(center, dtype=float),
                                    normal=np.asarray(primitive.normal, dtype=float),
                                    rotated_az=rotated_az,
                                    site_ids_local=list(site_ids_local),
                                    mol_class=str(mol_class),
                                    base_height=None,
                                )
                                if candidate is not None:
                                    site_candidates.append(candidate)
                            if prof is not None:
                                prof["n_height_none"] += 1
                            continue
                        for hidx, hdelta in enumerate(height_shift_list):
                            h_use = float(h + hdelta)
                            if h_use < self.config.min_height - 1e-12 or h_use > self.config.max_height + 1e-12:
                                continue
                            candidate = self._build_pose_candidate(
                                primitive_index=int(pidx),
                                basis_id=primitive.basis_id,
                                rotation_index=int(ridx),
                                azimuth_index=int(aidx),
                                azimuth_rad=float(az),
                                height_shift_index=int(hidx),
                                height_shift_delta=float(hdelta),
                                shift_uv=np.asarray(shift_uv, dtype=float),
                                quaternion=np.asarray(q_tot, dtype=float),
                                height=float(h_use),
                                center=np.asarray(center, dtype=float),
                                normal=np.asarray(primitive.normal, dtype=float),
                                rotated_adsorbate=rotated_az,
                                site_ids_local=list(site_ids_local),
                                mol_class=str(mol_class),
                            )
                            if candidate is not None:
                                site_candidates.append(candidate)
            site_candidates.sort(key=lambda x: x.height)
            if self.config.max_poses_per_site is not None:
                site_candidates = self._select_site_candidates(site_candidates, int(self.config.max_poses_per_site))
            candidates.extend(site_candidates)
        candidates.sort(key=lambda x: x.height)
        if prof is not None:
            prof["n_candidates_before_prune"] = int(len(candidates))
        t_prune0 = perf_counter() if prof is not None else 0.0
        kept = self._prune_initial_poses(candidates, slab)
        if prof is not None:
            prof["t_prune_s"] = float(perf_counter() - t_prune0)
            prof["n_candidates_after_prune"] = int(len(kept))
            self.last_profile = dict(prof)
        else:
            self.last_profile = {}
        self._prof = None
        self._nl = None
        self._surface_slab = None
        self._surface_slab_map = None
        self._surface_local_ids = None
        self._surface_r = None
        self._surface_pos = None
        self._surface_frac = None
        self._cell = None
        self._inv_cell = None
        self._pbc = None
        self._cell_orthogonal = False
        self._cell_lengths = None
        self._ads_r = None
        self._denom2_surface = None
        self._mic_rcell = None
        self._mic_inv_rcell = None
        self._mic_vrvecs = None
        return kept

    def _build_pose_candidate(
        self,
        *,
        primitive_index: int,
        basis_id: int | None,
        rotation_index: int,
        azimuth_index: int,
        azimuth_rad: float,
        height_shift_index: int,
        height_shift_delta: float,
        shift_uv: np.ndarray,
        quaternion: np.ndarray,
        height: float,
        center: np.ndarray,
        normal: np.ndarray,
        rotated_adsorbate: Atoms,
        site_ids_local: list[int],
        mol_class: str,
    ) -> PoseCandidate | None:
        translation = np.asarray(center, dtype=float) + float(height) * np.asarray(normal, dtype=float)
        placed_pos = np.asarray(rotated_adsorbate.positions, dtype=float) + translation
        prof = self._prof
        t1 = perf_counter() if prof is not None else 0.0
        if prof is not None:
            prof["n_has_clash"] += 1
        clash = self._check_clash_positions(placed_pos, float(self.config.clash_tau))
        if prof is not None:
            prof["t_has_clash_s"] += float(perf_counter() - t1)
        if clash:
            if bool(self.config.adaptive_height_fallback):
                return self._build_adaptive_height_candidate(
                    primitive_index=int(primitive_index),
                    basis_id=basis_id,
                    rotation_index=int(rotation_index),
                    azimuth_index=int(azimuth_index),
                    azimuth_rad=float(azimuth_rad),
                    shift_uv=np.asarray(shift_uv, dtype=float),
                    quaternion=np.asarray(quaternion, dtype=float),
                    center=np.asarray(center, dtype=float),
                    normal=np.asarray(normal, dtype=float),
                    rotated_az=rotated_adsorbate,
                    site_ids_local=list(site_ids_local),
                    mol_class=str(mol_class),
                    base_height=float(height),
                )
            return None
        placed = rotated_adsorbate.copy()
        placed.positions = placed_pos
        candidate = PoseCandidate(
            primitive_index=int(primitive_index),
            basis_id=basis_id,
            rotation_index=int(rotation_index),
            azimuth_index=int(azimuth_index),
            azimuth_rad=float(azimuth_rad),
            height_shift_index=int(height_shift_index),
            height_shift_delta=float(height_shift_delta),
            tilt_deg=self._estimate_tilt_deg(rotated_adsorbate, np.asarray(normal, dtype=float)),
            shift_uv=np.asarray(shift_uv, dtype=float),
            quaternion=np.asarray(quaternion, dtype=float),
            height=float(height),
            com=np.asarray(placed.get_center_of_mass(), dtype=float),
            atoms=placed,
        )
        if prof is not None:
            prof["n_candidates_raw"] += 1
        return candidate

    def _build_adaptive_height_candidate(
        self,
        *,
        primitive_index: int,
        basis_id: int | None,
        rotation_index: int,
        azimuth_index: int,
        azimuth_rad: float,
        shift_uv: np.ndarray,
        quaternion: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        rotated_az: Atoms,
        site_ids_local: list[int],
        mol_class: str,
        base_height: float | None,
    ) -> PoseCandidate | None:
        step = float(max(1e-4, self.config.adaptive_height_fallback_step, self.config.height_step))
        start = self._adaptive_height_start(
            base_height=base_height,
            mol_class=str(mol_class),
            site_coordination=len(site_ids_local),
        )
        stop = float(max(start, self.config.max_height) + max(0.0, float(self.config.adaptive_height_fallback_max_extra)))
        enforce_contact_window = len(site_ids_local) < 3
        extra_tol = 0.0
        if mol_class == "diatomic":
            extra_tol = float(self.config.diatomic_contact_extra_tolerance)
        elif mol_class == "linear":
            extra_tol = float(self.config.linear_contact_extra_tolerance)
        target_tau = float(max(float(self.config.clash_tau), float(np.max(np.asarray(self.config.height_taus, dtype=float)))))
        contact_limit = float(target_tau + self.config.site_contact_tolerance + extra_tol + self.config.adaptive_height_fallback_contact_slack)
        probe = start
        while probe <= stop + 1e-12:
            translation = np.asarray(center, dtype=float) + float(probe) * np.asarray(normal, dtype=float)
            placed_pos = np.asarray(rotated_az.positions, dtype=float) + translation
            if not self._check_clash_positions(placed_pos, float(self.config.clash_tau)):
                if not enforce_contact_window:
                    return self._build_pose_candidate(
                        primitive_index=int(primitive_index),
                        basis_id=basis_id,
                        rotation_index=int(rotation_index),
                        azimuth_index=int(azimuth_index),
                        azimuth_rad=float(azimuth_rad),
                        height_shift_index=-1,
                        height_shift_delta=float(0.0 if base_height is None else probe - float(base_height)),
                        shift_uv=np.asarray(shift_uv, dtype=float),
                        quaternion=np.asarray(quaternion, dtype=float),
                        height=float(probe),
                        center=np.asarray(center, dtype=float),
                        normal=np.asarray(normal, dtype=float),
                        rotated_adsorbate=rotated_az,
                        site_ids_local=list(site_ids_local),
                        mol_class=str(mol_class),
                    )
                min_site, min_surface = self._min_scaled_distance_site_and_surface(placed_pos, list(site_ids_local))
                contact_metric = self._site_contact_metric(
                    min_site=float(min_site),
                    min_surface=float(min_surface),
                    site_ids_local=list(site_ids_local),
                )
                if float(contact_metric) <= float(contact_limit):
                    return self._build_pose_candidate(
                        primitive_index=int(primitive_index),
                        basis_id=basis_id,
                        rotation_index=int(rotation_index),
                        azimuth_index=int(azimuth_index),
                        azimuth_rad=float(azimuth_rad),
                        height_shift_index=-1,
                        height_shift_delta=float(0.0 if base_height is None else probe - float(base_height)),
                        shift_uv=np.asarray(shift_uv, dtype=float),
                        quaternion=np.asarray(quaternion, dtype=float),
                        height=float(probe),
                        center=np.asarray(center, dtype=float),
                        normal=np.asarray(normal, dtype=float),
                        rotated_adsorbate=rotated_az,
                        site_ids_local=list(site_ids_local),
                        mol_class=str(mol_class),
                    )
            probe += step
        return None

    def _adaptive_height_start(self, *, base_height: float | None, mol_class: str, site_coordination: int) -> float:
        preferred = float(max(self.config.min_height, self._preferred_min_height(mol_class=mol_class, site_coordination=site_coordination)))
        if base_height is None:
            return preferred
        return float(max(preferred, float(base_height) + max(1e-4, float(self.config.adaptive_height_fallback_step))))

    def _center_adsorbate_for_sampling(self, adsorbate: Atoms) -> np.ndarray:
        ads = adsorbate.copy()
        mode = str(getattr(self.config, "placement_mode", "anchor_free")).strip().lower()
        if mode in {"anchor", "anchor_aware", "origin_atom"}:
            origin_index = self._select_adsorption_origin_index(adsorbate)
            return np.asarray(ads.positions, dtype=float) - np.asarray(ads.positions[int(origin_index)], dtype=float)
        ref_mode = str(getattr(self.config, "anchor_free_reference", "center_of_mass")).strip().lower()
        pos = np.asarray(ads.positions, dtype=float)
        if ref_mode in {"geom", "geometry", "geometric_center", "centroid"}:
            ref = np.mean(pos, axis=0, keepdims=False)
        else:
            ref = np.asarray(ads.get_center_of_mass(), dtype=float)
        return pos - np.asarray(ref, dtype=float)

    def _solve_height(
        self,
        slab: Atoms,
        ads_positions: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        mol_class: str,
        site_ids_local: list[int],
    ) -> float | None:
        lo = float(max(self.config.min_height, self._preferred_min_height(mol_class=mol_class, site_coordination=len(site_ids_local))))
        hi = float(self.config.max_height)
        step = float(max(1e-4, self.config.height_step))
        extra_tol = 0.0
        if mol_class == "diatomic":
            extra_tol = float(self.config.diatomic_contact_extra_tolerance)
        elif mol_class == "linear":
            extra_tol = float(self.config.linear_contact_extra_tolerance)
        for tau in self.config.height_taus:
            target_tau = float(max(tau, self.config.clash_tau))
            enforce_contact_window = len(site_ids_local) < 3
            probe_height = lo
            safe_height: float | None = None
            clash_height = lo - step
            while probe_height <= hi + 1e-12:
                ok, min_site, min_surface = self._check_height_constraints(
                    ads_positions=ads_positions,
                    center=center,
                    normal=normal,
                    height=probe_height,
                    target_tau=target_tau,
                    site_ids_local=site_ids_local,
                )
                if not ok:
                    clash_height = probe_height
                    probe_height += step
                    continue
                if probe_height <= lo + 1e-12:
                    return float(probe_height)
                safe_height = probe_height
                if enforce_contact_window:
                    contact_metric = self._site_contact_metric(
                        min_site=min_site,
                        min_surface=min_surface,
                        site_ids_local=site_ids_local,
                    )
                    if contact_metric > target_tau + self.config.site_contact_tolerance + extra_tol:
                        safe_height = None
                break
            if safe_height is None:
                continue
            low = max(lo, clash_height)
            high = safe_height
            for _ in range(max(0, int(self.config.height_bisect_steps))):
                mid = 0.5 * (low + high)
                ok, _, _ = self._check_height_constraints(
                    ads_positions=ads_positions,
                    center=center,
                    normal=normal,
                    height=mid,
                    target_tau=target_tau,
                    site_ids_local=site_ids_local,
                )
                if ok:
                    high = mid
                else:
                    low = mid
            _, min_site, min_surface = self._check_height_constraints(
                ads_positions=ads_positions,
                center=center,
                normal=normal,
                height=high,
                target_tau=target_tau,
                site_ids_local=site_ids_local,
            )
            if not enforce_contact_window:
                return float(high)
            contact_metric = self._site_contact_metric(
                min_site=min_site,
                min_surface=min_surface,
                site_ids_local=site_ids_local,
            )
            if contact_metric <= target_tau + self.config.site_contact_tolerance + extra_tol:
                return float(high)
        return None

    @staticmethod
    def _preferred_min_height(mol_class: str, site_coordination: int) -> float:
        # Linear adsorbates are more sensitive to overly low initial heights on
        # bridge/hollow sites. Use a conservative floor so relaxations do not
        # immediately collapse into a neighboring lower-coordination basin.
        if mol_class not in {"diatomic", "linear"}:
            return 0.0
        coord = max(1, int(site_coordination))
        if coord <= 1:
            return 1.75
        if coord == 2:
            return 1.55
        return 1.45

    @staticmethod
    def _site_contact_metric(min_site: float, min_surface: float, site_ids_local: list[int]) -> float:
        # Multi-center sites such as 3c/4c hollows should be judged by the closest
        # surface contact at the site center, not by forcing one specific site atom
        # to sit at a top-like distance.
        if len(site_ids_local) >= 3:
            return float(min_surface)
        return float(min_site)

    def _check_height_constraints(
        self,
        ads_positions: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        height: float,
        target_tau: float,
        site_ids_local: list[int],
    ) -> tuple[bool, float, float]:
        prof = self._prof
        t0 = perf_counter() if prof is not None else 0.0
        if prof is not None:
            prof["n_height_checks"] += 1
        translation = center + float(height) * normal
        placed_pos = np.asarray(ads_positions, dtype=float) + translation
        min_site, min_surface = self._min_scaled_distance_site_and_surface(placed_pos, site_ids_local)
        ok = bool(min_site >= target_tau and min_surface >= self.config.clash_tau)
        if prof is not None:
            prof["t_height_checks_s"] += float(perf_counter() - t0)
        return ok, float(min_site), float(min_surface)

    def _check_clash_positions(self, placed_pos: np.ndarray, tau: float) -> bool:
        _, min_surface = self._min_scaled_distance_site_and_surface(placed_pos, site_ids_local=[])
        return bool(min_surface < float(tau))

    def _min_scaled_distance_site_and_surface(self, placed_pos: np.ndarray, site_ids_local: list[int]) -> tuple[float, float]:
        surf_pos = self._surface_pos
        surf_frac = self._surface_frac
        inv_cell = self._inv_cell
        cell = self._cell
        pbc = self._pbc
        lengths = self._cell_lengths
        denom2 = self._denom2_surface
        ads_r = self._ads_r
        if surf_pos is None or denom2 is None or ads_r is None:
            return np.inf, np.inf
        na = int(len(placed_pos))
        ns = int(len(surf_pos))
        if na == 0 or ns == 0:
            return np.inf, np.inf
        if self._cell_orthogonal and inv_cell is not None and surf_frac is not None and lengths is not None and pbc is not None:
            kernel = self._ortho_min_ratio_kernel
            if kernel is not None:
                site_arr = np.asarray(site_ids_local, dtype=np.int64)
                min_site2, min_surface2 = kernel(
                    np.asarray(placed_pos, dtype=float),
                    np.asarray(surf_frac, dtype=float),
                    np.asarray(inv_cell, dtype=float),
                    np.asarray(lengths, dtype=float),
                    np.asarray(pbc, dtype=np.bool_),
                    np.asarray(denom2, dtype=float),
                    site_arr,
                )
                min_surface = float(np.sqrt(max(0.0, float(min_surface2))))
                if len(site_ids_local) == 0:
                    return np.inf, min_surface
                min_site = float(np.sqrt(max(0.0, float(min_site2))))
                return min_site, min_surface
            ads_frac = np.asarray(placed_pos @ inv_cell, dtype=float)
            df = surf_frac[None, :, :] - ads_frac[:, None, :]
            for ax in range(3):
                if bool(pbc[ax]):
                    df[:, :, ax] = df[:, :, ax] - np.round(df[:, :, ax])
            d0 = df[:, :, 0] * float(lengths[0])
            d1 = df[:, :, 1] * float(lengths[1])
            d2 = df[:, :, 2] * float(lengths[2])
            dist2 = d0 * d0 + d1 * d1 + d2 * d2
        elif cell is not None and pbc is not None:
            delta = surf_pos[None, :, :] - np.asarray(placed_pos, dtype=float)[:, None, :]
            delta_flat = delta.reshape(-1, 3)
            if self._mic_rcell is not None and self._mic_inv_rcell is not None and self._mic_vrvecs is not None:
                dist2 = self._mic_minlen2_cached(delta_flat).reshape(na, ns)
            else:
                _, d = find_mic(delta_flat, cell, pbc=pbc)
                dist2 = np.asarray(d, dtype=float).reshape(na, ns) ** 2
        else:
            delta = surf_pos[None, :, :] - np.asarray(placed_pos, dtype=float)[:, None, :]
            dist2 = np.sum(delta * delta, axis=2)
        min_s2 = float(np.min(dist2 / denom2))
        min_surface = float(np.sqrt(max(0.0, min_s2)))
        if len(site_ids_local) == 0:
            return np.inf, min_surface
        cols = np.asarray(site_ids_local, dtype=int)
        denom2_site = denom2[:, cols]
        dist2_site = dist2[:, cols]
        min_site = float(np.sqrt(max(0.0, float(np.min(dist2_site / denom2_site)))))
        return min_site, min_surface

    @staticmethod
    def _is_cell_orthogonal(cell: np.ndarray, tol: float = 1e-10) -> bool:
        c = np.asarray(cell, dtype=float).reshape(3, 3)
        a, b, d = c[0], c[1], c[2]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        nd = float(np.linalg.norm(d))
        if na < 1e-14 or nb < 1e-14 or nd < 1e-14:
            return False
        ab = abs(float(np.dot(a, b))) / (na * nb)
        ad = abs(float(np.dot(a, d))) / (na * nd)
        bd = abs(float(np.dot(b, d))) / (nb * nd)
        return bool(ab < tol and ad < tol and bd < tol)

    def _prepare_mic_cache(self) -> None:
        self._mic_rcell = None
        self._mic_inv_rcell = None
        self._mic_vrvecs = None
        cell = self._cell
        pbc = self._pbc
        if cell is None or pbc is None:
            return
        if self._cell_orthogonal:
            return
        dim = int(np.sum(np.asarray(pbc, dtype=bool)))
        if dim <= 0:
            return
        try:
            cell_full = np.asarray(complete_cell(cell), dtype=float)
            rcell, _ = minkowski_reduce(cell_full, pbc=pbc)
            rcell = np.asarray(rcell, dtype=float)
            inv_rcell = np.asarray(np.linalg.inv(rcell), dtype=float)
        except Exception:
            return
        ranges = [np.arange(-1 * int(bool(pp)), int(bool(pp)) + 1) for pp in pbc]
        hkls = np.array([(0, 0, 0)] + list(itertools.product(*ranges)), dtype=int)
        vrvecs = np.asarray(hkls @ rcell, dtype=float)
        self._mic_rcell = rcell
        self._mic_inv_rcell = inv_rcell
        self._mic_vrvecs = vrvecs

    def _mic_minlen2_cached(self, v: np.ndarray, chunk_size: int = 200000) -> np.ndarray:
        rcell = self._mic_rcell
        inv_rcell = self._mic_inv_rcell
        vrvecs = self._mic_vrvecs
        pbc = self._pbc
        if rcell is None or inv_rcell is None or vrvecs is None or pbc is None:
            raise ValueError("MIC cache not initialized.")
        vv = np.asarray(v, dtype=float)
        if vv.ndim == 1:
            vv = vv.reshape(1, 3)
        out = np.empty((vv.shape[0],), dtype=float)
        n = int(vv.shape[0])
        pbc_mask = np.asarray(pbc, dtype=bool)
        for start in range(0, n, int(chunk_size)):
            stop = min(n, start + int(chunk_size))
            chunk = vv[start:stop]
            frac = np.asarray(chunk @ inv_rcell, dtype=float)
            for ax in range(3):
                if bool(pbc_mask[ax]):
                    frac[:, ax] = frac[:, ax] - np.floor(frac[:, ax])
            pos = np.asarray(frac @ rcell, dtype=float)
            min2 = np.full((pos.shape[0],), np.inf, dtype=float)
            for t in vrvecs:
                d = pos + t
                d2 = np.sum(d * d, axis=1)
                min2 = np.minimum(min2, d2)
            out[start:stop] = min2
        return out

    @staticmethod
    def _sample_tangent_shifts(rng: np.random.Generator, n: int, radius: float) -> list[np.ndarray]:
        if n <= 1 or radius <= 0:
            return [np.zeros(2, dtype=float)]
        shifts: list[np.ndarray] = [np.zeros(2, dtype=float)]
        for _ in range(n - 1):
            ang = 2.0 * np.pi * rng.random()
            rr = float(radius) * np.sqrt(rng.random())
            shifts.append(np.array([rr * np.cos(ang), rr * np.sin(ang)], dtype=float))
        return shifts

    @staticmethod
    def _sample_uniform_quaternion(rng: np.random.Generator) -> np.ndarray:
        u1 = rng.random()
        u2 = rng.random()
        u3 = rng.random()
        q = np.array(
            [
                np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
                np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
                np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
                np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
            ],
            dtype=float,
        )
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _sample_azimuth_angles(n: int) -> list[float]:
        if n <= 1:
            return [0.0]
        return [float(v) for v in np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)]

    @staticmethod
    def _sample_height_shifts(n: int, step: float) -> list[float]:
        if n <= 1 or step <= 0:
            return [0.0]
        vals = [0.0]
        k = 1
        while len(vals) < n:
            vals.append(float(k * step))
            if len(vals) < n:
                vals.append(float(-k * step))
            k += 1
        return vals

    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    def _rotated_adsorbate(self, adsorbate_centered: Atoms, q: np.ndarray) -> Atoms:
        rot = self._quaternion_to_matrix(q)
        out = adsorbate_centered.copy()
        out.positions = np.asarray(adsorbate_centered.positions, dtype=float) @ rot.T
        return out

    @staticmethod
    def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        a = np.asarray(axis, dtype=float)
        n = np.linalg.norm(a)
        if n < 1e-12:
            return np.eye(3, dtype=float)
        a = a / n
        x, y, z = a
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=float,
        )

    def _rotate_around_axis(self, adsorbate_centered: Atoms, axis: np.ndarray, angle: float) -> Atoms:
        rot = self._axis_angle_to_matrix(axis, angle)
        out = adsorbate_centered.copy()
        out.positions = np.asarray(adsorbate_centered.positions, dtype=float) @ rot.T
        return out

    @staticmethod
    def _rotation_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
        a = np.asarray(axis, dtype=float)
        n = np.linalg.norm(a)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        a = a / n
        half = 0.5 * float(angle)
        s = float(np.sin(half))
        return np.array([a[0] * s, a[1] * s, a[2] * s, float(np.cos(half))], dtype=float)

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = np.asarray(q1, dtype=float)
        x2, y2, z2, w2 = np.asarray(q2, dtype=float)
        q = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _has_clash(slab: Atoms, adsorbate: Atoms, surface_atom_ids: list[int], tau: float) -> bool:
        if len(surface_atom_ids) == 0 or len(adsorbate) == 0:
            return False
        combined = slab + adsorbate
        n_slab = len(slab)
        ads_numbers = adsorbate.get_atomic_numbers()
        surf_numbers = slab.get_atomic_numbers()[surface_atom_ids]
        r_surf = np.asarray([covalent_radii[int(z)] for z in surf_numbers], dtype=float)
        for i, z_ads in enumerate(ads_numbers):
            r_ads = float(covalent_radii[int(z_ads)])
            ci = n_slab + i
            d = np.asarray(combined.get_distances(ci, surface_atom_ids, mic=True), dtype=float).reshape(-1)
            cutoff = float(tau) * (r_ads + r_surf)
            if bool(np.any(d < cutoff)):
                return True
        return False

    def _resolve_neighborlist_tau_factor(self) -> float:
        base = float(self.config.clash_tau)
        try:
            base = max(base, float(np.max(np.asarray(self.config.height_taus, dtype=float))))
        except Exception:
            base = float(self.config.clash_tau)
        extra = float(max(float(self.config.site_contact_tolerance), float(self.config.diatomic_contact_extra_tolerance), float(self.config.linear_contact_extra_tolerance)))
        return float(base + extra)

    def _setup_neighborlist(self, slab: Atoms, adsorbate: Atoms) -> None:
        self._nl = None
        if not bool(self.config.neighborlist_enabled):
            return
        if len(slab) < int(self.config.neighborlist_min_surface_atoms):
            return
        if len(slab) <= 0:
            return
        r_surf = self._surface_r
        if r_surf is None:
            z_surf = slab.get_atomic_numbers()
            r_surf = np.asarray([covalent_radii[int(z)] for z in z_surf], dtype=float)
        pad = float(self.config.neighborlist_cutoff_padding)
        tau = float(self._nl_tau_factor)
        surf_cutoffs = tau * np.asarray(r_surf, dtype=float) + pad
        z_ads = adsorbate.get_atomic_numbers()
        ads_cutoffs = np.asarray([tau * float(covalent_radii[int(z)]) + pad for z in z_ads], dtype=float)
        cutoffs = np.concatenate([surf_cutoffs, ads_cutoffs], dtype=float)
        self._nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True)

    def _min_scaled_distance_nl(self, combined: Atoms, n_slab: int, adsorbate: Atoms, slab_atom_ids: list[int]) -> float:
        if len(slab_atom_ids) == 0:
            return np.inf
        nl = self._nl
        r_slab = self._surface_r
        if nl is None or r_slab is None:
            return np.inf
        allow = set(int(i) for i in slab_atom_ids)
        ads_numbers = adsorbate.get_atomic_numbers()
        min_ratio = np.inf
        for i, z_ads in enumerate(ads_numbers):
            r_ads = float(covalent_radii[int(z_ads)])
            ci = int(n_slab + i)
            nbs, _ = nl.get_neighbors(ci)
            if len(nbs) == 0:
                continue
            keep = [int(j) for j in nbs if int(j) < n_slab and int(j) in allow]
            if not keep:
                continue
            d = np.asarray(combined.get_distances(ci, keep, mic=True), dtype=float).reshape(-1)
            if d.size == 0:
                continue
            denom = np.maximum(1e-12, r_ads + np.asarray(r_slab[np.asarray(keep, dtype=int)], dtype=float))
            ratio_min = float(np.min(d / denom))
            if ratio_min < min_ratio:
                min_ratio = ratio_min
        return float(min_ratio)

    def _has_clash_nl(self, slab: Atoms, adsorbate: Atoms, tau: float) -> bool:
        nl = self._nl
        r_slab = self._surface_r
        if nl is None or r_slab is None:
            return False
        if len(slab) == 0 or len(adsorbate) == 0:
            return False
        combined = slab + adsorbate
        n_slab = int(len(slab))
        nl.update(combined)
        ads_numbers = adsorbate.get_atomic_numbers()
        for i, z_ads in enumerate(ads_numbers):
            r_ads = float(covalent_radii[int(z_ads)])
            ci = int(n_slab + i)
            nbs, _ = nl.get_neighbors(ci)
            if len(nbs) == 0:
                continue
            keep = [int(j) for j in nbs if int(j) < n_slab]
            if not keep:
                continue
            d = np.asarray(combined.get_distances(ci, keep, mic=True), dtype=float).reshape(-1)
            if d.size == 0:
                continue
            cutoff = float(tau) * (r_ads + np.asarray(r_slab[np.asarray(keep, dtype=int)], dtype=float))
            if bool(np.any(d < cutoff)):
                return True
        return False

    def _map_site_atom_ids_to_surface(self, site_atom_ids: tuple[int, ...]) -> list[int] | None:
        if self._surface_slab_map is None:
            return None
        mapped: list[int] = []
        for sid in site_atom_ids:
            key = int(sid)
            if key not in self._surface_slab_map:
                return None
            mapped.append(int(self._surface_slab_map[key]))
        return mapped

    @staticmethod
    def _min_scaled_distance_from_combined(combined: Atoms, n_slab: int, ads_numbers: np.ndarray, slab_atom_ids: list[int]) -> float:
        if len(slab_atom_ids) == 0:
            return np.inf
        surf_numbers = combined.get_atomic_numbers()[slab_atom_ids]
        r_surf = np.asarray([covalent_radii[int(z)] for z in surf_numbers], dtype=float)
        min_ratio = np.inf
        for i, z_ads in enumerate(ads_numbers):
            r_ads = float(covalent_radii[int(z_ads)])
            ci = int(n_slab + i)
            d = np.asarray(combined.get_distances(ci, slab_atom_ids, mic=True), dtype=float).reshape(-1)
            if d.size == 0:
                continue
            denom = np.maximum(1e-12, (r_ads + r_surf))
            ratio_min = float(np.min(d / denom))
            if ratio_min < min_ratio:
                min_ratio = ratio_min
        return float(min_ratio)

    @staticmethod
    def _min_scaled_distance(slab: Atoms, adsorbate: Atoms, surface_atom_ids: list[int]) -> float:
        if len(surface_atom_ids) == 0:
            return np.inf
        combined = slab + adsorbate
        n_slab = len(slab)
        ads_numbers = adsorbate.get_atomic_numbers()
        surf_numbers = slab.get_atomic_numbers()[surface_atom_ids]
        r_surf = np.asarray([covalent_radii[int(z)] for z in surf_numbers], dtype=float)
        min_ratio = np.inf
        for i, z_ads in enumerate(ads_numbers):
            r_ads = float(covalent_radii[int(z_ads)])
            ci = n_slab + i
            d = np.asarray(combined.get_distances(ci, surface_atom_ids, mic=True), dtype=float).reshape(-1)
            denom = np.maximum(1e-12, (r_ads + r_surf))
            ratio_min = float(np.min(d / denom)) if d.size > 0 else float("inf")
            if ratio_min < min_ratio:
                min_ratio = ratio_min
        return float(min_ratio)

    def _prune_initial_poses(self, candidates: list[PoseCandidate], slab: Atoms) -> list[PoseCandidate]:
        if len(candidates) <= 1:
            return candidates
        kept: list[PoseCandidate] = []
        for cand in candidates:
            is_dup = False
            for ref in kept:
                d_com = self._mic_point_distance(cand.com, ref.com, slab)
                d_rot = self._quaternion_distance(cand.quaternion, ref.quaternion)
                if d_com < self.config.prune_com_distance and d_rot < self.config.prune_rot_distance:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
        return kept

    @staticmethod
    def _select_site_candidates(site_candidates: list[PoseCandidate], max_keep: int) -> list[PoseCandidate]:
        if max_keep <= 0 or len(site_candidates) <= max_keep:
            return site_candidates[: max_keep] if max_keep > 0 else []
        buckets: dict[tuple[int, int], list[PoseCandidate]] = {}
        for cand in site_candidates:
            tilt_bin = 0
            if cand.tilt_deg >= 60.0:
                tilt_bin = 2
            elif cand.tilt_deg >= 30.0:
                tilt_bin = 1
            key = (int(cand.azimuth_index), int(cand.rotation_index), int(tilt_bin))
            buckets.setdefault(key, []).append(cand)
        for arr in buckets.values():
            arr.sort(key=lambda x: x.height)
        ordered_keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))
        selected: list[PoseCandidate] = []
        while len(selected) < max_keep:
            changed = False
            for k in ordered_keys:
                if buckets[k]:
                    selected.append(buckets[k].pop(0))
                    changed = True
                    if len(selected) >= max_keep:
                        break
            if not changed:
                break
        selected.sort(key=lambda x: x.height)
        return selected

    def _effective_sampling_budget(self, mol_class: str) -> tuple[int, int, int]:
        n_rot = max(1, int(self.config.n_rotations))
        n_az = max(1, int(self.config.n_azimuth))
        n_shift = max(1, int(self.config.n_shifts))
        if mol_class == "monatomic":
            return 1, 1, n_shift
        if mol_class in {"diatomic", "linear"}:
            # Keep the user-requested angular budget for linear adsorbates.
            # Reducing it here made default CO/N2 sampling collapse almost entirely
            # to upright poses, which contradicts the plan's SE(3) sampling intent.
            return max(2, n_rot), n_az, n_shift
        return n_rot, n_az, n_shift

    @staticmethod
    def _select_adsorption_origin_index(adsorbate: Atoms) -> int:
        return PoseSampler._select_binding_atom_index(adsorbate)

    @staticmethod
    def _select_binding_atom_index(adsorbate: Atoms) -> int:
        n = int(len(adsorbate))
        if n <= 1:
            return 0
        adj = PoseSampler._build_bond_adjacency(adsorbate)
        z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
        non_h = [i for i, zi in enumerate(z) if int(zi) != 1]
        if len(non_h) == 1:
            return int(non_h[0])
        pos = np.asarray(adsorbate.get_positions(), dtype=float)
        center = np.mean(pos, axis=0)
        degrees = [len(v) for v in adj]
        hetero = {7, 8, 15, 16}

        def has_h_neighbor(i: int) -> bool:
            return any(int(z[j]) == 1 for j in adj[i])

        def has_c_like_neighbor(i: int) -> bool:
            return any(int(z[j]) in {6, 14} for j in adj[i])

        def low_en_score(i: int) -> tuple[float, float, float, int]:
            zi = int(z[i])
            en = float(_PAULING_ELECTRONEGATIVITY.get(zi, 10.0))
            radial = float(np.linalg.norm(pos[i] - center))
            return (en, -radial, float(covalent_radii[zi]), int(i))

        def hetero_score(i: int) -> tuple[float, int, float, int]:
            zi = int(z[i])
            en = float(_PAULING_ELECTRONEGATIVITY.get(zi, 0.0))
            radial = float(np.linalg.norm(pos[i] - center))
            return (en, -int(degrees[i]), radial, -int(i))

        heavy = non_h or list(range(n))
        protic_hetero = [i for i in heavy if int(z[i]) in hetero and has_h_neighbor(i)]
        if protic_hetero:
            return int(max(protic_hetero, key=hetero_score))

        hetero_c_like = [i for i in heavy if int(z[i]) in hetero and has_c_like_neighbor(i) and int(degrees[i]) <= 2]
        if hetero_c_like and len(heavy) > 2:
            return int(max(hetero_c_like, key=hetero_score))

        terminal_heavy = [i for i in heavy if int(degrees[i]) <= 1]
        candidates = terminal_heavy or heavy
        best = min(candidates, key=low_en_score)
        return int(best)

    @staticmethod
    def orient_adsorbate_for_binding(
        adsorbate: Atoms,
        binding_atom_index: int | None = None,
        normal: np.ndarray | None = None,
    ) -> Atoms:
        idx = PoseSampler._select_binding_atom_index(adsorbate) if binding_atom_index is None else int(binding_atom_index)
        out = adsorbate.copy()
        pos = np.asarray(out.get_positions(), dtype=float)
        if len(pos) == 0:
            return out
        anchor = np.asarray(pos[idx], dtype=float)
        centered = pos - anchor
        target = np.array([0.0, 0.0, 1.0], dtype=float) if normal is None else np.asarray(normal, dtype=float)
        target_norm = float(np.linalg.norm(target))
        if target_norm < 1e-12:
            target = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            target = target / target_norm
        other_ids = [i for i in range(len(out)) if int(i) != idx]
        if other_ids:
            ref = PoseSampler._best_frame_direction(centered[other_ids])
            if float(np.linalg.norm(ref)) < 1e-12:
                axis = PoseSampler._principal_axis(out)
                ref = np.asarray(axis if axis is not None else target, dtype=float)
            quat = PoseSampler._quaternion_from_two_vectors(ref, target)
            if quat is not None:
                rot = PoseSampler._quaternion_to_matrix(quat)
                centered = centered @ rot.T
        out.positions = centered
        out.info["binding_atom_index"] = int(idx)
        out.info["binding_atom_symbol"] = str(out[int(idx)].symbol)
        return out

    @staticmethod
    def _build_bond_adjacency(adsorbate: Atoms, bond_tau: float = 1.20) -> list[set[int]]:
        n = int(len(adsorbate))
        adj: list[set[int]] = [set() for _ in range(n)]
        if n <= 1:
            return adj
        z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
        d = np.asarray(adsorbate.get_all_distances(mic=False), dtype=float)
        for i in range(n):
            ri = float(covalent_radii[int(z[i])])
            for j in range(i + 1, n):
                rj = float(covalent_radii[int(z[j])])
                if float(d[i, j]) <= float(bond_tau) * (ri + rj):
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def _build_site_oriented_quaternions(
        self,
        adsorbate_centered: Atoms,
        normal: np.ndarray,
        mol_class: str,
        rng: np.random.Generator,
        n_rot: int,
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        if mol_class in {"diatomic", "linear"}:
            axis = self._principal_axis(adsorbate_centered)
            if axis is not None:
                normal_u = np.asarray(normal, dtype=float)
                normal_u = normal_u / (np.linalg.norm(normal_u) + 1e-12)
                tangent = self._reference_tangent(normal_u)
                entries: list[tuple[float, int, np.ndarray]] = []
                for tilt_deg in self._linear_tilt_schedule(n_rot):
                    theta = np.deg2rad(float(tilt_deg))
                    target_base = np.cos(theta) * normal_u + np.sin(theta) * tangent
                    target_base = target_base / (np.linalg.norm(target_base) + 1e-12)
                    for sign_idx, target in enumerate((target_base, -target_base)):
                        q = self._quaternion_from_two_vectors(axis, target)
                        if q is not None:
                            qn = np.asarray(q, dtype=float)
                            duplicate = any(self._quaternion_distance(qn, ref_q) <= 1e-6 for _, _, ref_q in entries)
                            if not duplicate:
                                entries.append((float(tilt_deg), int(sign_idx), qn))
                if entries:
                    return self._select_linear_quaternion_entries(entries, int(n_rot))
        elif mol_class == "nonlinear":
            normal_u = np.asarray(normal, dtype=float)
            normal_u = normal_u / (np.linalg.norm(normal_u) + 1e-12)
            if bool(getattr(self.config, "nonlinear_atom_down_coverage", True)):
                max_vectors = max(1, int(getattr(self.config, "nonlinear_atom_down_max_vectors", 4)))
                for q_down in self._nonlinear_atom_down_quaternions(
                    adsorbate_centered,
                    max_vectors=min(int(n_rot), int(max_vectors)),
                    normal=normal_u,
                ):
                    self._append_unique_quaternion(out, q_down)
                    if len(out) >= n_rot:
                        return out[:n_rot]
            q_align = self._body_frame_alignment_quaternion(adsorbate_centered)
            for q_body in self._body_frame_euler_schedule(n_rot):
                q_use = np.asarray(q_body, dtype=float)
                if q_align is not None:
                    q_use = self._quaternion_multiply(q_body, q_align)
                self._append_unique_quaternion(out, q_use)
                if len(out) >= n_rot:
                    return out[:n_rot]
        while len(out) < n_rot:
            self._append_unique_quaternion(out, self._sample_uniform_quaternion(rng))
        return out[:n_rot]

    def _append_unique_quaternion(self, acc: list[np.ndarray], quat: np.ndarray, tol: float = 1e-6) -> None:
        q = np.asarray(quat, dtype=float)
        q = q / (np.linalg.norm(q) + 1e-12)
        for ref in acc:
            if self._quaternion_distance(q, ref) <= float(tol):
                return
        acc.append(q)

    @staticmethod
    def _estimate_tilt_deg(adsorbate_centered: Atoms, normal: np.ndarray) -> float:
        axis = PoseSampler._principal_axis(adsorbate_centered)
        n = np.asarray(normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        if axis is None:
            return 0.0
        c = abs(float(np.dot(axis, n)))
        c = min(1.0, max(-1.0, c))
        return float(np.degrees(np.arccos(c)))

    @staticmethod
    def _principal_axis(adsorbate_centered: Atoms) -> np.ndarray | None:
        axes = PoseSampler._principal_axes(adsorbate_centered)
        if axes is None:
            return None
        axis = np.asarray(axes[:, 0], dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return None
        return axis / n

    @staticmethod
    def _principal_axes(adsorbate_centered: Atoms) -> np.ndarray | None:
        pos = np.asarray(adsorbate_centered.get_positions(), dtype=float)
        if len(pos) < 2:
            return None
        masses = np.asarray(adsorbate_centered.get_masses(), dtype=float).reshape(-1)
        mass_sum = float(np.sum(masses))
        if mass_sum <= 1e-12:
            masses = np.ones((len(pos),), dtype=float)
            mass_sum = float(len(pos))
        center = np.sum(pos * masses[:, None], axis=0, keepdims=True) / mass_sum
        centered = pos - center
        inertia = np.zeros((3, 3), dtype=float)
        for m, r in zip(masses, centered, strict=False):
            rr = float(np.dot(r, r))
            inertia += float(m) * (rr * np.eye(3, dtype=float) - np.outer(r, r))
        try:
            vals, vecs = np.linalg.eigh(inertia)
        except np.linalg.LinAlgError:
            return None
        order = np.argsort(np.asarray(vals, dtype=float))
        axes = np.asarray(vecs[:, order], dtype=float)
        # Canonicalize the sign of each axis to keep quaternion sampling reproducible.
        for k in range(axes.shape[1]):
            axis = axes[:, k]
            projections = centered @ axis
            idx = int(np.argmax(np.abs(projections)))
            if idx < len(projections) and float(projections[idx]) < 0.0:
                axes[:, k] = -axis
        if float(np.linalg.det(axes)) < 0.0:
            axes[:, -1] = -axes[:, -1]
        return axes

    @staticmethod
    def _body_frame_alignment_quaternion(adsorbate_centered: Atoms) -> np.ndarray | None:
        axes = PoseSampler._principal_axes(adsorbate_centered)
        if axes is None:
            return None
        rot = np.asarray(axes.T, dtype=float)
        return PoseSampler._quaternion_from_matrix(rot)

    @staticmethod
    def _nonlinear_atom_down_quaternions(
        adsorbate_centered: Atoms,
        *,
        max_vectors: int,
        normal: np.ndarray,
    ) -> list[np.ndarray]:
        pos = np.asarray(adsorbate_centered.get_positions(), dtype=float)
        if len(pos) <= 1 or max_vectors <= 0:
            return []
        z = np.asarray(adsorbate_centered.get_atomic_numbers(), dtype=int)
        candidate_ids = [
            int(i)
            for i, zi in enumerate(z)
            if int(zi) != 1 and float(np.linalg.norm(pos[i])) > 1e-8
        ]
        if not candidate_ids:
            candidate_ids = [int(i) for i in range(len(pos)) if float(np.linalg.norm(pos[i])) > 1e-8]
        scored = []
        for idx in candidate_ids:
            zi = int(z[idx])
            en = float(_PAULING_ELECTRONEGATIVITY.get(zi, 0.0))
            radial = float(np.linalg.norm(pos[idx]))
            is_hetero = int(zi in {7, 8, 9, 15, 16, 17, 35, 53})
            scored.append((int(-is_hetero), -en, -radial, int(idx)))
        scored.sort()
        out: list[np.ndarray] = []
        for _, _, _, idx in scored:
            other_ids = [int(i) for i in range(len(pos)) if int(i) != int(idx)]
            if not other_ids:
                continue
            ref = PoseSampler._best_frame_direction(pos[other_ids] - np.asarray(pos[int(idx)], dtype=float))
            n = float(np.linalg.norm(ref))
            if n < 1e-12:
                continue
            q = PoseSampler._quaternion_from_two_vectors(ref, np.asarray(normal, dtype=float))
            if q is None:
                continue
            duplicate = any(PoseSampler._quaternion_distance(np.asarray(q, dtype=float), ref_q) <= 1e-6 for ref_q in out)
            if duplicate:
                continue
            out.append(np.asarray(q, dtype=float))
            if len(out) >= int(max_vectors):
                break
        return [np.asarray(q, dtype=float) for q in out]

    @staticmethod
    def _best_frame_direction(rel_vectors: np.ndarray) -> np.ndarray:
        rel = np.asarray(rel_vectors, dtype=float)
        if rel.ndim != 2 or rel.shape[0] == 0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        norms = np.linalg.norm(rel, axis=1)
        keep = norms > 1e-12
        if not np.any(keep):
            return np.array([0.0, 0.0, 1.0], dtype=float)
        rel_u = rel[keep] / norms[keep][:, None]
        candidates: list[np.ndarray] = []
        for i in range(len(rel_u)):
            candidates.append(np.asarray(rel_u[i], dtype=float))
        for i in range(len(rel_u)):
            for j in range(i + 1, len(rel_u)):
                cand = np.asarray(rel_u[i] + rel_u[j], dtype=float)
                if float(np.linalg.norm(cand)) > 1e-12:
                    candidates.append(cand / (np.linalg.norm(cand) + 1e-12))
        mean_vec = np.mean(rel_u, axis=0)
        if float(np.linalg.norm(mean_vec)) > 1e-12:
            candidates.append(mean_vec / (np.linalg.norm(mean_vec) + 1e-12))
        if not candidates:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        best = None
        for cand in candidates:
            c = np.asarray(cand, dtype=float)
            c = c / (np.linalg.norm(c) + 1e-12)
            score = float(np.min(rel_u @ c))
            rank = (-score, -float(np.mean(rel_u @ c)))
            if best is None or rank < best[0]:
                best = (rank, c)
        return np.asarray(best[1], dtype=float) if best is not None else np.array([0.0, 0.0, 1.0], dtype=float)

    @staticmethod
    def _body_frame_euler_schedule(n_rot: int) -> list[np.ndarray]:
        n = max(1, int(n_rot))
        out: list[np.ndarray] = [np.array([0.0, 0.0, 0.0, 1.0], dtype=float)]
        if n <= 1:
            return out
        phi1 = 0.6180339887498948
        phi2 = 0.4142135623730950
        for k in range(1, n):
            u = (float(k) + 0.5) / float(n)
            alpha = 2.0 * np.pi * ((float(k) * phi1) % 1.0)
            beta = float(np.arccos(np.clip(1.0 - 2.0 * u, -1.0, 1.0)))
            gamma = 2.0 * np.pi * ((float(k) * phi2) % 1.0)
            rz1 = PoseSampler._axis_angle_to_matrix(np.array([0.0, 0.0, 1.0], dtype=float), alpha)
            ry = PoseSampler._axis_angle_to_matrix(np.array([0.0, 1.0, 0.0], dtype=float), beta)
            rz2 = PoseSampler._axis_angle_to_matrix(np.array([0.0, 0.0, 1.0], dtype=float), gamma)
            rot = rz1 @ ry @ rz2
            quat = PoseSampler._quaternion_from_matrix(rot)
            if quat is not None:
                out.append(quat)
        return out[:n]

    @staticmethod
    def _reference_tangent(normal: np.ndarray) -> np.ndarray:
        n = np.asarray(normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, n))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        t = ref - float(np.dot(ref, n)) * n
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            t = ref - float(np.dot(ref, n)) * n
            nt = np.linalg.norm(t)
        return t / (nt + 1e-12)

    @staticmethod
    def _linear_tilt_schedule(n_rot: int) -> list[float]:
        n = max(1, int(n_rot))
        base = [0.0, 25.0, 50.0, 80.0]
        if n <= len(base):
            return base[:n]
        extra = np.linspace(base[-1], 88.0, num=(n - len(base) + 1), endpoint=True)[1:]
        return base + [float(v) for v in extra.tolist()]

    @staticmethod
    def _select_linear_quaternion_entries(entries: list[tuple[float, int, np.ndarray]], n_rot: int) -> list[np.ndarray]:
        if n_rot <= 0:
            return []
        if len(entries) <= n_rot:
            return [np.asarray(q, dtype=float) for _, _, q in entries]
        by_tilt: dict[float, dict[int, np.ndarray]] = {}
        for tilt_deg, sign_idx, quat in entries:
            by_tilt.setdefault(float(tilt_deg), {})[int(sign_idx)] = np.asarray(quat, dtype=float)
        tilts = sorted(by_tilt)
        ordered: list[np.ndarray] = []
        for pass_idx in range(2):
            for tilt_idx, tilt_deg in enumerate(tilts):
                sign_pref = int((tilt_idx + pass_idx) % 2)
                group = by_tilt[float(tilt_deg)]
                for sign_idx in (sign_pref, 1 - sign_pref):
                    quat = group.get(int(sign_idx))
                    if quat is None:
                        continue
                    duplicate = any(PoseSampler._quaternion_distance(np.asarray(quat, dtype=float), ref) <= 1e-6 for ref in ordered)
                    if duplicate:
                        continue
                    ordered.append(np.asarray(quat, dtype=float))
                    break
                if len(ordered) >= int(n_rot):
                    return ordered[: int(n_rot)]
        for _, _, quat in entries:
            duplicate = any(PoseSampler._quaternion_distance(np.asarray(quat, dtype=float), ref) <= 1e-6 for ref in ordered)
            if duplicate:
                continue
            ordered.append(np.asarray(quat, dtype=float))
            if len(ordered) >= int(n_rot):
                break
        return ordered[: int(n_rot)]

    @staticmethod
    def _quaternion_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray | None:
        a = np.asarray(v_from, dtype=float)
        b = np.asarray(v_to, dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return None
        a = a / na
        b = b / nb
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot > 1.0 - 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if dot < -1.0 + 1e-10:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(a, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            axis = np.cross(a, ref)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            return np.array([axis[0], axis[1], axis[2], 0.0], dtype=float)
        axis = np.cross(a, b)
        q = np.array([axis[0], axis[1], axis[2], 1.0 + dot], dtype=float)
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _quaternion_from_matrix(rot: np.ndarray) -> np.ndarray | None:
        r = np.asarray(rot, dtype=float).reshape(3, 3)
        trace = float(np.trace(r))
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            if s < 1e-12:
                return None
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        else:
            diag = np.diag(r)
            idx = int(np.argmax(diag))
            if idx == 0:
                s = 2.0 * np.sqrt(max(1e-16, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
                x = 0.25 * s
                y = (r[0, 1] + r[1, 0]) / s
                z = (r[0, 2] + r[2, 0]) / s
                w = (r[2, 1] - r[1, 2]) / s
            elif idx == 1:
                s = 2.0 * np.sqrt(max(1e-16, 1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
                x = (r[0, 1] + r[1, 0]) / s
                y = 0.25 * s
                z = (r[1, 2] + r[2, 1]) / s
                w = (r[0, 2] - r[2, 0]) / s
            else:
                s = 2.0 * np.sqrt(max(1e-16, 1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
                x = (r[0, 2] + r[2, 0]) / s
                y = (r[1, 2] + r[2, 1]) / s
                z = 0.25 * s
                w = (r[1, 0] - r[0, 1]) / s
        q = np.array([x, y, z, w], dtype=float)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return None
        return q / n

    @staticmethod
    def _classify_molecule_shape(adsorbate: Atoms) -> str:
        n = len(adsorbate)
        if n <= 1:
            return "monatomic"
        if n == 2:
            return "diatomic"
        pos = np.asarray(adsorbate.get_positions(), dtype=float)
        pos = pos - np.mean(pos, axis=0, keepdims=True)
        cov = pos.T @ pos
        evals = np.sort(np.linalg.eigvalsh(cov))
        if evals[-1] <= 1e-12:
            return "monatomic"
        if evals[1] / evals[-1] < 1e-3:
            return "linear"
        return "nonlinear"

    @staticmethod
    def _quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        a = np.asarray(q1, dtype=float)
        b = np.asarray(q2, dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return float(np.inf)
        a = a / na
        b = b / nb
        dot = float(np.dot(a, b))
        dot = abs(dot)
        if dot >= 1.0 - 1e-12:
            return 0.0
        dot = min(1.0, max(-1.0, dot))
        return 2.0 * float(np.arccos(dot))

    @staticmethod
    def _mic_point_distance(p1: np.ndarray, p2: np.ndarray, slab: Atoms) -> float:
        cell = np.asarray(slab.cell.array, dtype=float)
        pbc = slab.get_pbc()
        diff = p1 - p2
        try:
            frac = np.linalg.solve(cell.T, diff)
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(diff))
        for i in range(3):
            if pbc[i]:
                frac[i] -= np.round(frac[i])
        dcart = frac @ cell
        return float(np.linalg.norm(dcart))
