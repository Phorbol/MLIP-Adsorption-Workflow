from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

from adsorption_ensemble.site.primitives import SitePrimitive
from adsorption_ensemble.surface.pipeline import SurfaceContext


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _unit_cell_corners_xy(slab: Atoms) -> np.ndarray:
    cell_xy = slab.cell[:2, :2]
    return np.array([[0.0, 0.0], cell_xy[0, :], cell_xy[0, :] + cell_xy[1, :], cell_xy[1, :], [0.0, 0.0]])


def plot_surface_sites_from_groups(
    slab: Atoms,
    groups: dict[str, list[np.ndarray]],
    surface_indices: list[int],
    filename: str | Path,
    atom_radius_scale: float = 500.0,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 8))
    all_pos = slab.get_positions()
    z = all_pos[:, 2]
    plot_order = np.argsort(z)
    atomic_numbers = slab.get_atomic_numbers()
    corners = _unit_cell_corners_xy(slab)
    ax.plot(corners[:, 0], corners[:, 1], "k--", label="Unit Cell")
    surface_set = set(surface_indices)
    sub_surface_indices = [i for i in plot_order if i not in surface_set]
    if sub_surface_indices:
        sub_pos = all_pos[sub_surface_indices]
        sub_colors = [jmol_colors[atomic_numbers[i]] for i in sub_surface_indices]
        sub_radii = [covalent_radii[atomic_numbers[i]] for i in sub_surface_indices]
        ax.scatter(sub_pos[:, 0], sub_pos[:, 1], s=np.array(sub_radii) ** 2 * atom_radius_scale, c=sub_colors, alpha=0.25, edgecolors="none", label="Sub-surface")
    if surface_indices:
        surf_idx_sorted = sorted(surface_indices, key=lambda i: z[i])
        surf_pos = all_pos[surf_idx_sorted]
        surf_colors = [jmol_colors[atomic_numbers[i]] for i in surf_idx_sorted]
        surf_radii = [covalent_radii[atomic_numbers[i]] for i in surf_idx_sorted]
        ax.scatter(surf_pos[:, 0], surf_pos[:, 1], s=np.array(surf_radii) ** 2 * atom_radius_scale, c=surf_colors, alpha=0.85, edgecolors="black", linewidths=0.2, label="Surface")
        for i in surf_idx_sorted:
            ax.text(all_pos[i, 0], all_pos[i, 1], str(i), ha="center", va="center", fontsize=7, color="black")
    marker_map = {"1c": "x", "2c": "P", "3c": "^", "4c": "D"}
    cmap = plt.get_cmap("tab20")
    gi = 0
    for kind, grouped_centers in groups.items():
        marker = marker_map.get(kind, "o")
        for centers in grouped_centers:
            c = np.array(centers, dtype=float)
            if c.size == 0:
                continue
            color = cmap(gi % 20)
            ax.scatter(c[:, 0], c[:, 1], s=100, c=[color], marker=marker, linewidths=1.8, label=f"{kind}-g{gi}")
            gi += 1
    ax.set_aspect("equal")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title("Surface Atoms and Site Groups")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    ax.set_xlim(corners[:, 0].min() - 0.2, corners[:, 0].max() + 0.2)
    ax.set_ylim(corners[:, 1].min() - 0.2, corners[:, 1].max() + 0.2)
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_surface_primitives_2d(
    slab: Atoms,
    context: SurfaceContext,
    primitives: list[SitePrimitive],
    filename: str | Path,
    atom_radius_scale: float = 500.0,
) -> Path:
    grouped: dict[str, list[np.ndarray]] = {"1c": [], "2c": [], "3c": [], "4c": []}
    topo_group: dict[tuple[str, str], list[np.ndarray]] = {}
    for p in primitives:
        key = (p.kind, p.topo_hash)
        topo_group.setdefault(key, []).append(np.asarray(p.center, dtype=float))
    for (kind, _), centers in topo_group.items():
        grouped.setdefault(kind, []).append(np.array(centers, dtype=float))
    return plot_surface_sites_from_groups(
        slab=slab,
        groups=grouped,
        surface_indices=context.detection.surface_atom_ids,
        filename=filename,
        atom_radius_scale=atom_radius_scale,
    )


def plot_site_centers_only(
    slab: Atoms,
    primitives: list[SitePrimitive],
    filename: str | Path,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 8))
    corners = _unit_cell_corners_xy(slab)
    ax.plot(corners[:, 0], corners[:, 1], "k--", label="Unit Cell")
    kind_style = {
        "1c": dict(marker="x", color="#1f77b4"),
        "2c": dict(marker="+", color="#aec7e8"),
        "3c": dict(marker="^", color="#ff7f0e"),
        "4c": dict(marker="D", color="#ffbb78"),
    }
    by_kind: dict[str, list[np.ndarray]] = {"1c": [], "2c": [], "3c": [], "4c": []}
    for p in primitives:
        by_kind.setdefault(p.kind, []).append(p.center)
    for k, pts in by_kind.items():
        if not pts:
            continue
        arr = np.array(pts)
        sty = kind_style.get(k, dict(marker="o", color="black"))
        ax.scatter(arr[:, 0], arr[:, 1], s=55, marker=sty["marker"], c=sty["color"], label=k)
    ax.set_aspect("equal")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title("Site Centers Only")
    ax.legend(loc="upper right", fontsize="small")
    ax.set_xlim(corners[:, 0].min() - 0.2, corners[:, 0].max() + 0.2)
    ax.set_ylim(corners[:, 1].min() - 0.2, corners[:, 1].max() + 0.2)
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_inequivalent_sites_2d(
    slab: Atoms,
    primitives: list[SitePrimitive],
    filename: str | Path,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 8))
    corners = _unit_cell_corners_xy(slab)
    ax.plot(corners[:, 0], corners[:, 1], "k--", label="Unit Cell")
    kind_style = {
        "1c": dict(marker="x", color="#1f77b4"),
        "2c": dict(marker="+", color="#aec7e8"),
        "3c": dict(marker="^", color="#ff7f0e"),
        "4c": dict(marker="D", color="#ffbb78"),
    }
    by_kind: dict[str, list[np.ndarray]] = {"1c": [], "2c": [], "3c": [], "4c": []}
    reps: dict[int, SitePrimitive] = {}
    for p in primitives:
        by_kind.setdefault(p.kind, []).append(np.asarray(p.center, dtype=float))
        bid = int(p.basis_id) if p.basis_id is not None else -1
        if bid not in reps:
            reps[bid] = p
    for k, pts in by_kind.items():
        if not pts:
            continue
        arr = np.array(pts, dtype=float)
        sty = kind_style.get(k, dict(marker="o", color="black"))
        ax.scatter(arr[:, 0], arr[:, 1], s=25, marker=sty["marker"], c=sty["color"], alpha=0.18, linewidths=0.5)
    for bid, p in sorted(reps.items(), key=lambda x: x[0]):
        sty = kind_style.get(p.kind, dict(marker="o", color="black"))
        c = np.asarray(p.center, dtype=float)
        ax.scatter([c[0]], [c[1]], s=120, marker=sty["marker"], c=sty["color"], edgecolors="black", linewidths=0.8, label=f"{p.kind}-b{bid}")
        ax.text(c[0], c[1], f"b{bid}", fontsize=8, color="black", ha="left", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title("Inequivalent Site Representatives")
    ax.legend(loc="upper right", fontsize="x-small")
    ax.set_xlim(corners[:, 0].min() - 0.2, corners[:, 0].max() + 0.2)
    ax.set_ylim(corners[:, 1].min() - 0.2, corners[:, 1].max() + 0.2)
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_site_embedding_pca(
    primitives: list[SitePrimitive],
    filename: str | Path,
    title: str = "Site Embedding PCA (All vs Inequivalent)",
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 7))
    reps: dict[int, int] = {}
    feats = []
    basis_ids = []
    for i, p in enumerate(primitives):
        if p.embedding is None:
            continue
        v = np.asarray(p.embedding, dtype=float).reshape(-1)
        if v.size == 0 or not np.all(np.isfinite(v)):
            continue
        bid = int(p.basis_id) if p.basis_id is not None else -1
        if bid not in reps:
            reps[bid] = len(basis_ids)
        feats.append(v)
        basis_ids.append(bid)
    if not feats:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    x = np.asarray(feats, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    if x.shape[0] == 1 or x.shape[1] == 1:
        coords = np.zeros((x.shape[0], 2), dtype=float)
        evr = (0.0, 0.0)
    else:
        try:
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            comps = vt[:2].T
            coords = x @ comps
            denom = float(np.sum(s**2)) + 1e-12
            evr = (float((s[0] ** 2) / denom), float((s[1] ** 2) / denom)) if s.size > 1 else (float((s[0] ** 2) / denom), 0.0)
        except np.linalg.LinAlgError:
            coords = np.zeros((x.shape[0], 2), dtype=float)
            evr = (0.0, 0.0)
    coords = np.asarray(coords, dtype=float)
    ax.scatter(coords[:, 0], coords[:, 1], s=30, c="#7f7f7f", alpha=0.25, edgecolors="none", label="All sites")
    rep_points = []
    rep_labels = []
    for bid, idx in sorted(reps.items(), key=lambda t: t[0]):
        rep_points.append(coords[int(idx)])
        rep_labels.append(f"b{bid}")
    if rep_points:
        rp = np.asarray(rep_points, dtype=float)
        ax.scatter(rp[:, 0], rp[:, 1], s=160, marker="*", c="#d62728", edgecolors="black", linewidths=0.6, label="Inequivalent reps")
        for (x0, y0), lab in zip(rp, rep_labels):
            ax.text(float(x0), float(y0), lab, fontsize=8, color="black", ha="left", va="bottom")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
