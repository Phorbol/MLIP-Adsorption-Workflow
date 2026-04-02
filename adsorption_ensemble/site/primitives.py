from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from ase import Atom, Atoms
from ase.build import add_adsorbate

from adsorption_ensemble.surface.graph import ExposedSurfaceGraph
from adsorption_ensemble.surface.pipeline import SurfaceContext


@dataclass
class SitePrimitive:
    kind: str
    atom_ids: tuple[int, ...]
    center: np.ndarray
    normal: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    topo_hash: str
    basis_id: int | None = None
    embedding: np.ndarray | None = None
    site_label: str | None = None


class PrimitiveEnumerator:
    def enumerate(self, graph: ExposedSurfaceGraph) -> dict[str, list[tuple[int, ...]]]:
        nodes = sorted(graph.surface_atom_ids)
        nbrs = graph.neighbors
        p1 = [tuple([i]) for i in nodes]
        p2 = sorted(graph.edges)
        p3_set: set[tuple[int, int, int]] = set()
        for i, j in p2:
            common = nbrs[i].intersection(nbrs[j])
            for k in common:
                tri = tuple(sorted((i, j, int(k))))
                if len(set(tri)) == 3:
                    p3_set.add(tri)
        p3 = sorted(p3_set)
        p4_set: set[tuple[int, int, int, int]] = set()
        for a in nodes:
            for b, c in combinations(sorted(nbrs[a]), 2):
                if c in nbrs[b]:
                    continue
                for d in nbrs[b].intersection(nbrs[c]):
                    if d == a:
                        continue
                    quad = tuple(sorted((a, b, c, int(d))))
                    if len(set(quad)) != 4:
                        continue
                    if self._is_chordless_cycle4(quad, nbrs):
                        p4_set.add(quad)
        p4 = sorted(p4_set)
        return {"1c": p1, "2c": p2, "3c": p3, "4c": p4}

    @staticmethod
    def _is_chordless_cycle4(quad: tuple[int, int, int, int], nbrs: dict[int, set[int]]) -> bool:
        nodes = list(quad)
        all_edges = {(min(i, j), max(i, j)) for i, j in combinations(nodes, 2) if j in nbrs[i]}
        if len(all_edges) < 4:
            return False
        deg = {n: 0 for n in nodes}
        for i, j in all_edges:
            deg[i] += 1
            deg[j] += 1
        return all(v == 2 for v in deg.values())


class LocalFrameBuilder:
    def build(self, slab: Atoms, atom_ids: tuple[int, ...], global_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center, positions_unwrapped = self._center_and_unwrapped_positions(slab, atom_ids)
        normal = self._build_normal(positions_unwrapped, global_normal)
        t1 = self._build_t1(positions_unwrapped, center, normal)
        t2 = np.cross(normal, t1)
        t2_norm = np.linalg.norm(t2)
        if t2_norm < 1e-10:
            t2 = self._fallback_t2(normal, t1)
        else:
            t2 = t2 / t2_norm
        return center, normal, t1, t2

    @staticmethod
    def _center_and_unwrapped_positions(slab: Atoms, atom_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        ref = int(atom_ids[0])
        ref_pos = slab.positions[ref]
        vectors = [np.zeros(3, dtype=float)]
        for j in atom_ids[1:]:
            v = slab.get_distances(ref, int(j), mic=True, vector=True)
            vectors.append(np.asarray(v, dtype=float).reshape(3))
        vectors_arr = np.array(vectors)
        positions_unwrapped = ref_pos + vectors_arr
        center = np.mean(positions_unwrapped, axis=0)
        center_frac = np.linalg.solve(slab.cell.array.T, center)
        pbc = slab.get_pbc()
        for ax in range(3):
            if pbc[ax]:
                center_frac[ax] = center_frac[ax] % 1.0
        center = center_frac @ slab.cell.array
        return center, positions_unwrapped

    @staticmethod
    def _build_normal(positions: np.ndarray, global_normal: np.ndarray) -> np.ndarray:
        g = global_normal / (np.linalg.norm(global_normal) + 1e-12)
        if positions.shape[0] < 3:
            return g
        centered = positions - np.mean(positions, axis=0)
        try:
            _, _, vh = np.linalg.svd(centered)
            n = vh[-1]
        except np.linalg.LinAlgError:
            return g
        n = n / (np.linalg.norm(n) + 1e-12)
        if np.dot(n, g) < 0:
            n = -n
        return n

    @staticmethod
    def _build_t1(positions: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
        for p in positions:
            v = p - center
            v = v - np.dot(v, normal) * normal
            n = np.linalg.norm(v)
            if n > 1e-10:
                return v / n
        return LocalFrameBuilder._fallback_t1(normal)

    @staticmethod
    def _fallback_t1(normal: np.ndarray) -> np.ndarray:
        basis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(basis, normal)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        v = basis - np.dot(basis, normal) * normal
        return v / (np.linalg.norm(v) + 1e-12)

    @staticmethod
    def _fallback_t2(normal: np.ndarray, t1: np.ndarray) -> np.ndarray:
        t2 = np.cross(normal, t1)
        n = np.linalg.norm(t2)
        if n < 1e-10:
            basis = np.array([0.0, 0.0, 1.0])
            t2 = np.cross(normal, basis)
            n = np.linalg.norm(t2)
        return t2 / (n + 1e-12)


class PrimitiveBuilder:
    def __init__(
        self,
        enumerator: PrimitiveEnumerator | None = None,
        frame_builder: LocalFrameBuilder | None = None,
        min_site_distance: float = 0.1,
        min_polygon_area: float = 0.05,
        min_center_surface_distance: float = 0.20,
        center_surface_distance_factor: float = 0.35,
        use_ase_reference_sites: bool = True,
    ):
        self.enumerator = enumerator or PrimitiveEnumerator()
        self.frame_builder = frame_builder or LocalFrameBuilder()
        self.min_site_distance = min_site_distance
        self.min_polygon_area = min_polygon_area
        self.min_center_surface_distance = min_center_surface_distance
        self.center_surface_distance_factor = center_surface_distance_factor
        self.use_ase_reference_sites = use_ase_reference_sites

    def build(self, slab: Atoms, context: SurfaceContext) -> list[SitePrimitive]:
        normal_axis = context.classification.normal_axis
        if normal_axis is None:
            raise ValueError("SurfaceContext does not have a valid normal axis.")
        global_normal = np.zeros(3, dtype=float)
        global_normal[normal_axis] = 1.0
        groups = self.enumerator.enumerate(context.graph)
        primitives: list[SitePrimitive] = []
        for kind in ("1c", "2c", "3c", "4c"):
            for atom_ids in groups[kind]:
                center, normal, t1, t2 = self.frame_builder.build(slab=slab, atom_ids=atom_ids, global_normal=global_normal)
                topo_hash = self._make_topo_hash(kind, atom_ids, context.graph)
                primitives.append(
                    SitePrimitive(
                        kind=kind,
                        atom_ids=atom_ids,
                        center=center,
                        normal=normal,
                        t1=t1,
                        t2=t2,
                        topo_hash=topo_hash,
                        site_label=None,
                    )
                )
        primitives = self._prune_by_geometry(
            primitives,
            slab,
            context.classification.normal_axis,
            context.detection.surface_atom_ids,
        )
        primitives = self._prune_by_center_distance(primitives, slab)
        if self.use_ase_reference_sites:
            ref = self._build_from_ase_reference_sites(
                slab=slab,
                normal_axis=int(context.classification.normal_axis),
                surface_ids=list(context.detection.surface_atom_ids),
                graph=context.graph,
            )
            if ref is not None and len(ref) > 0:
                return ref
        return primitives

    def _build_from_ase_reference_sites(
        self,
        slab: Atoms,
        normal_axis: int,
        surface_ids: list[int],
        graph: ExposedSurfaceGraph,
    ) -> list[SitePrimitive] | None:
        info = slab.info.get("adsorbate_info", {})
        if not isinstance(info, dict):
            return None
        sites = info.get("sites", {})
        if not isinstance(sites, dict) or not sites:
            return None
        global_normal = np.zeros(3, dtype=float)
        global_normal[int(normal_axis)] = 1.0
        out: list[SitePrimitive] = []
        for name in sorted(str(k) for k in sites.keys()):
            center = self._ase_site_center(slab, name)
            if center is None:
                continue
            atom_ids = self._nearest_surface_atoms_for_site_name(slab, center, surface_ids, name)
            if len(atom_ids) <= 0:
                continue
            c0, normal, t1, t2 = self.frame_builder.build(slab=slab, atom_ids=atom_ids, global_normal=global_normal)
            _ = c0
            kind = f"{len(atom_ids)}c"
            topo_hash = self._make_topo_hash(kind, atom_ids, graph)
            out.append(
                SitePrimitive(
                    kind=kind,
                    atom_ids=atom_ids,
                    center=np.asarray(center, dtype=float),
                    normal=normal,
                    t1=t1,
                    t2=t2,
                    topo_hash=topo_hash,
                    site_label=str(name).lower(),
                )
            )
        if not out:
            return None
        kind_order = {"1c": 0, "2c": 1, "3c": 2, "4c": 3}
        out.sort(key=lambda p: (kind_order.get(p.kind, 99), p.atom_ids))
        return out

    @staticmethod
    def _ase_site_center(slab: Atoms, site_name: str) -> np.ndarray | None:
        try:
            probe = slab.copy()
            add_adsorbate(probe, Atom("He"), 0.0, position=str(site_name))
            return np.asarray(probe.positions[-1], dtype=float)
        except Exception:
            return None

    def _nearest_surface_atoms_for_site_name(
        self,
        slab: Atoms,
        center: np.ndarray,
        surface_ids: list[int],
        site_name: str,
    ) -> tuple[int, ...]:
        lname = str(site_name).lower()
        if lname == "ontop":
            n = 1
        elif "bridge" in lname:
            n = 2
        elif lname in {"fcc", "hcp"}:
            n = 3
        elif lname == "hollow":
            n = self._infer_hollow_coordination(slab, center, surface_ids)
        else:
            n = 1
        return self._nearest_surface_atoms(slab=slab, center=center, surface_ids=surface_ids, n=n)

    def _infer_hollow_coordination(self, slab: Atoms, center: np.ndarray, surface_ids: list[int]) -> int:
        if len(surface_ids) < 4:
            return 3
        d = []
        for i in surface_ids:
            di = self._mic_point_distance(np.asarray(slab.positions[int(i)], dtype=float), np.asarray(center, dtype=float), slab)
            d.append(float(di))
        d = np.sort(np.asarray(d, dtype=float))
        if d.shape[0] < 4:
            return 3
        r = float((d[3] + 1e-12) / (d[2] + 1e-12))
        return 4 if r < 1.12 else 3

    def _nearest_surface_atoms(self, slab: Atoms, center: np.ndarray, surface_ids: list[int], n: int) -> tuple[int, ...]:
        if n <= 0 or not surface_ids:
            return tuple()
        ds = []
        for i in surface_ids:
            di = self._mic_point_distance(np.asarray(slab.positions[int(i)], dtype=float), np.asarray(center, dtype=float), slab)
            ds.append((float(di), int(i)))
        ds.sort(key=lambda x: (x[0], x[1]))
        return tuple(sorted(int(i) for _, i in ds[: int(n)]))

    @staticmethod
    def _make_topo_hash(kind: str, atom_ids: tuple[int, ...], graph: ExposedSurfaceGraph) -> str:
        deg = sorted(len(graph.neighbors[i]) for i in atom_ids)
        return f"{kind}|n={len(atom_ids)}|deg={','.join(map(str, deg))}"

    def _prune_by_geometry(
        self,
        primitives: list[SitePrimitive],
        slab: Atoms,
        normal_axis: int | None,
        surface_ids: list[int],
    ) -> list[SitePrimitive]:
        if normal_axis is None:
            return primitives
        tangential_axes = [ax for ax in range(3) if ax != normal_axis]
        positions = slab.get_positions()
        nn = self._surface_nearest_neighbor_distance(slab, surface_ids)
        dyn_center_cut = max(self.min_center_surface_distance, self.center_surface_distance_factor * nn) if np.isfinite(nn) else self.min_center_surface_distance
        kept: list[SitePrimitive] = []
        for p in primitives:
            if p.kind in {"3c", "4c"}:
                area = self._site_area(positions[list(p.atom_ids)], tangential_axes)
                if area < self.min_polygon_area:
                    continue
            if p.kind == "4c":
                max_dev = self._quad_max_angle_deviation_from_right(positions[list(p.atom_ids)], tangential_axes)
                if max_dev > 25.0:
                    continue
            if len(p.atom_ids) >= 2:
                dmin = self._point_to_surface_min_distance_mic(slab, p.center, surface_ids)
                if dmin < dyn_center_cut and len(p.atom_ids) >= 3:
                    continue
            kept.append(p)
        return kept

    @staticmethod
    def _site_area(points: np.ndarray, tangential_axes: list[int]) -> float:
        if points.shape[0] < 3:
            return 0.0
        xy = points[:, tangential_axes]
        c = np.mean(xy, axis=0)
        vec = xy - c
        ang = np.arctan2(vec[:, 1], vec[:, 0])
        order = np.argsort(ang)
        poly = xy[order]
        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(area)

    @staticmethod
    def _quad_max_angle_deviation_from_right(points: np.ndarray, tangential_axes: list[int]) -> float:
        if points.shape[0] != 4:
            return 180.0
        xy = points[:, tangential_axes]
        c = np.mean(xy, axis=0)
        vec = xy - c
        ang = np.arctan2(vec[:, 1], vec[:, 0])
        order = np.argsort(ang)
        poly = xy[order]
        max_dev = 0.0
        for i in range(4):
            p_prev = poly[(i - 1) % 4]
            p_now = poly[i]
            p_next = poly[(i + 1) % 4]
            v1 = p_prev - p_now
            v2 = p_next - p_now
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-12 or n2 < 1e-12:
                return 180.0
            cang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cang)))
            max_dev = max(max_dev, abs(angle_deg - 90.0))
        return float(max_dev)

    @staticmethod
    def _surface_nearest_neighbor_distance(slab: Atoms, surface_ids: list[int]) -> float:
        if len(surface_ids) < 2:
            return np.inf
        dmin = np.inf
        for i, a in enumerate(surface_ids):
            for b in surface_ids[i + 1 :]:
                d = slab.get_distance(int(a), int(b), mic=True)
                if d > 1e-8 and d < dmin:
                    dmin = d
        return float(dmin)

    @staticmethod
    def _point_to_surface_min_distance_mic(slab: Atoms, point: np.ndarray, surface_ids: list[int]) -> float:
        if len(surface_ids) == 0:
            return np.inf
        probe = slab.copy()
        probe.append(Atom("He", position=np.asarray(point, dtype=float)))
        pid = len(probe) - 1
        dmin = np.inf
        for i in surface_ids:
            d = probe.get_distance(pid, int(i), mic=True)
            if d < dmin:
                dmin = d
        return float(dmin)

    def _prune_by_center_distance(self, primitives: list[SitePrimitive], slab: Atoms) -> list[SitePrimitive]:
        if self.min_site_distance <= 0 or len(primitives) <= 1:
            return primitives
        order = sorted(
            range(len(primitives)),
            key=lambda i: (
                {"1c": 0, "2c": 1, "3c": 2, "4c": 3}.get(str(primitives[i].kind), 99),
                primitives[i].topo_hash,
                -len(primitives[i].atom_ids),
            ),
        )
        kept: list[SitePrimitive] = []
        for idx in order:
            p = primitives[idx]
            duplicate = False
            for q in kept:
                d = self._mic_point_distance(p.center, q.center, slab)
                if d < self.min_site_distance:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(p)
        kind_order = {"1c": 0, "2c": 1, "3c": 2, "4c": 3}
        kept.sort(key=lambda x: (kind_order.get(x.kind, 99), x.atom_ids))
        return kept

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
