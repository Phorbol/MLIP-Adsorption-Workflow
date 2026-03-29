from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from itertools import product

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs


@dataclass
class SurfaceDetectionResult:
    surface_atom_ids: list[int]
    exposure_scores: np.ndarray
    diagnostics: dict


class SurfaceAtomDetector(ABC):
    @abstractmethod
    def detect(self, atoms: Atoms, normal_axis: int) -> SurfaceDetectionResult:
        raise NotImplementedError


def mic_displacements(cell: np.ndarray, vecs: np.ndarray, pbc: tuple[bool, bool, bool]) -> np.ndarray:
    frac = np.linalg.solve(cell.T, vecs.T).T
    for i in range(3):
        if pbc[i]:
            frac[:, i] -= np.round(frac[:, i])
    return np.dot(frac, cell)


class ProbeScanDetector(SurfaceAtomDetector):
    def __init__(self, grid_step: float = 0.6, probe_radius: float = 0.6, margin: float = 2.5, scale: float = 1.2):
        self.grid_step = grid_step
        self.probe_radius = probe_radius
        self.margin = margin
        self.scale = scale

    def detect(self, atoms: Atoms, normal_axis: int) -> SurfaceDetectionResult:
        cell = np.asarray(atoms.cell.array, dtype=float)
        lengths = atoms.cell.lengths()
        if lengths[normal_axis] <= 1e-8:
            return SurfaceDetectionResult([], np.zeros(len(atoms), dtype=float), {"reason": "invalid_cell_axis"})
        tangential_axes = [ax for ax in range(3) if ax != normal_axis]
        pbc = [True, True, True]
        pbc[normal_axis] = False
        pbc_t = tuple(bool(x) for x in pbc)
        scaled = atoms.get_scaled_positions(wrap=True)
        positions = np.dot(scaled, atoms.cell.array)
        f_min = float(np.min(scaled[:, normal_axis]))
        f_max = float(np.max(scaled[:, normal_axis]))
        cutoffs = np.asarray(natural_cutoffs(atoms), dtype=float)
        step_u = min(0.25, max(0.02, self.grid_step / max(lengths[tangential_axes[0]], 1e-8)))
        step_v = min(0.25, max(0.02, self.grid_step / max(lengths[tangential_axes[1]], 1e-8)))
        step_n = min(0.20, max(0.01, self.grid_step / max(lengths[normal_axis], 1e-8)))
        margin_n = self.margin / max(lengths[normal_axis], 1e-8)
        grids = [np.arange(0.0, 1.0, step_u), np.arange(0.0, 1.0, step_v)]
        hits = np.zeros(len(atoms), dtype=int)
        rays = 0
        for g1, g2 in product(*grids):
            rays += 1
            probe_frac = np.zeros(3, dtype=float)
            probe_frac[tangential_axes[0]] = g1
            probe_frac[tangential_axes[1]] = g2
            fn = f_max + margin_n
            while fn > f_min - margin_n:
                probe_frac[normal_axis] = fn
                probe = probe_frac @ cell
                vecs = positions - probe
                vecs = mic_displacements(cell, vecs, pbc_t)
                dists = np.linalg.norm(vecs, axis=1)
                j = int(np.argmin(dists))
                if dists[j] < self.scale * (self.probe_radius + cutoffs[j]):
                    hits[j] += 1
                    break
                fn -= step_n
        ids = np.where(hits > 0)[0].tolist()
        scores = hits.astype(float)
        if scores.max(initial=0.0) > 0:
            scores = scores / scores.max()
        return SurfaceDetectionResult(ids, scores, {"method": "probe_scan", "normal_axis": normal_axis, "ray_count": rays})


class VoxelFloodDetector(SurfaceAtomDetector):
    def __init__(self, spacing: float = 0.8, occ_scale: float = 1.15):
        self.spacing = spacing
        self.occ_scale = occ_scale

    def detect(self, atoms: Atoms, normal_axis: int) -> SurfaceDetectionResult:
        lengths = atoms.cell.lengths()
        if np.any(lengths < 1e-8):
            return SurfaceDetectionResult([], np.zeros(len(atoms), dtype=float), {"reason": "invalid_cell"})
        dims = [max(4, int(np.ceil(lengths[i] / self.spacing)) + 1) for i in range(3)]
        grid = np.zeros(dims, dtype=np.uint8)
        cell = np.asarray(atoms.cell.array, dtype=float)
        scaled = atoms.get_scaled_positions(wrap=True)
        positions = scaled @ cell
        pbc = [True, True, True]
        pbc[normal_axis] = False
        cutoffs = np.asarray(natural_cutoffs(atoms), dtype=float) * self.occ_scale
        frac_step = np.asarray([1.0 / (dims[i] - 1) for i in range(3)], dtype=float)
        for i, frac_center in enumerate(scaled):
            center_idx = np.clip(np.round(frac_center / frac_step).astype(int), [0, 0, 0], np.array(dims, dtype=int) - 1)
            radius = max(cutoffs[i], self.spacing)
            half = np.ceil((radius / np.maximum(lengths, 1e-8)) * (np.array(dims) - 1)).astype(int)
            mins = np.maximum(center_idx - half, 0)
            maxs = np.minimum(center_idx + half + 1, dims)
            for ix in range(mins[0], maxs[0]):
                for iy in range(mins[1], maxs[1]):
                    for iz in range(mins[2], maxs[2]):
                        frac_pt = np.array([ix, iy, iz], dtype=float) * frac_step
                        dfrac = frac_pt - frac_center
                        for ax in range(3):
                            if pbc[ax]:
                                dfrac[ax] -= np.round(dfrac[ax])
                        dcart = dfrac @ cell
                        if np.linalg.norm(dcart) <= radius:
                            grid[ix, iy, iz] = 1
        exterior = np.zeros_like(grid, dtype=np.uint8)
        q: deque[tuple[int, int, int]] = deque()
        first_plane = 0
        last_plane = dims[normal_axis] - 1
        seeds = []
        for ix in range(dims[0]):
            for iy in range(dims[1]):
                idx0 = [ix, iy, 0]
                idx1 = [ix, iy, 0]
                idx0[normal_axis] = first_plane
                idx1[normal_axis] = last_plane
                seeds.append(tuple(idx0))
                seeds.append(tuple(idx1))
        for s in seeds:
            if grid[s] == 0 and exterior[s] == 0:
                exterior[s] = 1
                q.append(s)
        neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in neighbors:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= dims[0] or ny >= dims[1] or nz >= dims[2]:
                    continue
                if grid[nx, ny, nz] == 0 and exterior[nx, ny, nz] == 0:
                    exterior[nx, ny, nz] = 1
                    q.append((nx, ny, nz))
        scores = np.zeros(len(atoms), dtype=float)
        for i, frac_center in enumerate(scaled):
            idx = np.clip(np.round(frac_center / frac_step).astype(int), [0, 0, 0], np.array(dims, dtype=int) - 1)
            contact = 0
            for dx, dy, dz in neighbors:
                nx, ny, nz = idx[0] + dx, idx[1] + dy, idx[2] + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= dims[0] or ny >= dims[1] or nz >= dims[2]:
                    continue
                if exterior[nx, ny, nz] == 1:
                    contact += 1
            scores[i] = float(contact)
        ids = np.where(scores > 0)[0].tolist()
        diagnostics = {"method": "voxel_flood", "normal_axis": normal_axis, "grid_shape": tuple(dims)}
        if scores.max(initial=0.0) > 0:
            scores = scores / scores.max()
        return SurfaceDetectionResult(ids, scores, diagnostics)
