from __future__ import annotations

from pathlib import Path

import numpy as np


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_energy_delta_hist(
    energies_ev: np.ndarray,
    e_min_ev: float,
    delta_e_ev: float,
    filename: str | Path,
    title: str,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    e = np.asarray(energies_ev, dtype=float)
    ok = np.isfinite(e)
    e = e[ok]
    if e.size == 0:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    de = e - float(e_min_ev)
    de = de[np.isfinite(de)]
    if de.size == 0:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    bins = int(min(80, max(10, np.sqrt(de.size) * 5)))
    ax.hist(de, bins=bins, color="#4c78a8", alpha=0.65, edgecolor="white")
    ax.axvline(float(delta_e_ev), color="#d62728", linewidth=2.0, linestyle="--", label=f"ΔE cut = {delta_e_ev:.3g} eV")
    ax.set_xlabel("E - Emin (eV)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mindist_hist(
    mindist: np.ndarray,
    threshold: float,
    filename: str | Path,
    title: str,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    d = np.asarray(mindist, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    if d.size == 0:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    bins = int(min(80, max(10, np.sqrt(d.size) * 5)))
    ax.hist(d, bins=bins, color="#72b7b2", alpha=0.65, edgecolor="white")
    ax.axvline(float(threshold), color="#d62728", linewidth=2.0, linestyle="--", label=f"dist cut = {threshold:.3g}")
    ax.set_xlabel("Min distance to selected (feature space)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_deltae_vs_mindist(
    energies_ev: np.ndarray,
    e_min_ev: float,
    delta_e_ev: float,
    mindist: np.ndarray,
    rmsd_threshold: float,
    energy_keep_mask: np.ndarray,
    rmsd_keep_mask: np.ndarray,
    filename: str | Path,
    title: str,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 7))
    e = np.asarray(energies_ev, dtype=float)
    de = e - float(e_min_ev)
    md = np.asarray(mindist, dtype=float)
    energy_keep = np.asarray(energy_keep_mask, dtype=bool)
    rmsd_keep = np.asarray(rmsd_keep_mask, dtype=bool)
    n = int(max(len(de), len(md), len(energy_keep), len(rmsd_keep)))
    if n == 0:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    de = de[:n]
    md = md[:n]
    energy_keep = energy_keep[:n]
    rmsd_keep = rmsd_keep[:n]
    ok_e = np.isfinite(de)
    ok_md = np.isfinite(md)
    ax.axvline(float(delta_e_ev), color="#d62728", linewidth=2.0, linestyle="--")
    ax.axhline(float(rmsd_threshold), color="#d62728", linewidth=2.0, linestyle="--")
    dropped_energy = (~energy_keep) & ok_e
    ax.scatter(de[dropped_energy], np.zeros(np.sum(dropped_energy), dtype=float), s=22, c="#7f7f7f", alpha=0.30, label="Dropped by ΔE")
    cand = energy_keep & ok_e & ok_md
    dropped_rmsd = cand & (~rmsd_keep)
    kept = cand & rmsd_keep
    if np.any(dropped_rmsd):
        ax.scatter(de[dropped_rmsd], md[dropped_rmsd], s=30, c="#ff7f0e", alpha=0.55, label="Dropped by dist")
    if np.any(kept):
        ax.scatter(de[kept], md[kept], s=70, c="#d62728", alpha=0.85, edgecolors="black", linewidths=0.35, label="Kept")
    ax.set_xlabel("E - Emin (eV)")
    ax.set_ylabel("Min distance to selected (feature space)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_adsorption_energy_hist(
    adsorption_energies_ev: np.ndarray,
    filename: str | Path,
    title: str,
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    e = np.asarray(adsorption_energies_ev, dtype=float)
    ok = np.isfinite(e)
    e = e[ok]
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    if e.size == 0:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    bins = int(min(100, max(10, np.sqrt(e.size) * 6)))
    ax.hist(e, bins=bins, color="#4c78a8", alpha=0.70, edgecolor="white")
    ax.axvline(float(np.nanmedian(e)), color="#ff7f0e", linewidth=2.0, linestyle="--", label="Median")
    ax.set_xlabel("Adsorption energy (eV)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
