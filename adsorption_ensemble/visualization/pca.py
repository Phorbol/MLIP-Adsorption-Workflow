from __future__ import annotations

from pathlib import Path

import numpy as np


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_feature_pca_compare(
    features: np.ndarray,
    selected_ids: list[int] | tuple[int, ...],
    filename: str | Path,
    title: str,
    selected_label: str = "Selected",
    all_label: str = "All",
) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 7))
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.size == 0:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    ok_rows = np.all(np.isfinite(x), axis=1)
    x = x[ok_rows]
    if x.size == 0:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
    sel = [int(i) for i in selected_ids if int(i) >= 0 and int(i) < int(features.shape[0])]
    sel_mask = np.zeros(int(features.shape[0]), dtype=bool)
    if sel:
        sel_mask[np.asarray(sel, dtype=int)] = True
    sel_mask = sel_mask[ok_rows]
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
            evr = (
                float((s[0] ** 2) / denom),
                float((s[1] ** 2) / denom) if s.size > 1 else 0.0,
            )
        except np.linalg.LinAlgError:
            coords = np.zeros((x.shape[0], 2), dtype=float)
            evr = (0.0, 0.0)
    coords = np.asarray(coords, dtype=float)
    ax.scatter(coords[:, 0], coords[:, 1], s=28, c="#7f7f7f", alpha=0.25, edgecolors="none", label=all_label)
    if np.any(sel_mask):
        ax.scatter(
            coords[sel_mask, 0],
            coords[sel_mask, 1],
            s=90,
            marker="o",
            c="#d62728",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label=selected_label,
        )
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
