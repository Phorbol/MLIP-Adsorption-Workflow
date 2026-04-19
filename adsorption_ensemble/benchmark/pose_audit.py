from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from ase import Atoms

from adsorption_ensemble.pose import PoseSampler


def classify_tilt_bin(tilt_deg: float) -> str:
    tilt = float(tilt_deg)
    if not np.isfinite(tilt):
        return "unknown"
    if tilt < 10.0:
        return "upright"
    if tilt < 60.0:
        return "tilted"
    return "flat"


def summarize_pose_frames(
    frames: list[Atoms],
    *,
    slab_n: int,
    primitives: list[Any] | None = None,
) -> dict[str, Any]:
    primitive_normals: dict[int, np.ndarray] = {}
    primitive_meta: dict[int, dict[str, Any]] = {}
    for idx, primitive in enumerate(primitives or []):
        primitive_normals[int(idx)] = np.asarray(getattr(primitive, "normal", np.array([0.0, 0.0, 1.0], dtype=float)), dtype=float)
        primitive_meta[int(idx)] = {
            "basis_id": None if getattr(primitive, "basis_id", None) is None else int(primitive.basis_id),
            "site_kind": str(getattr(primitive, "kind", "")),
            "site_label": (
                str(getattr(primitive, "site_label", ""))
                if getattr(primitive, "site_label", None) is not None
                else None
            ),
        }

    site_rows: dict[str, dict[str, Any]] = {}
    orientation_counts: Counter[str] = Counter()
    rotation_counts: Counter[int] = Counter()
    azimuth_counts: Counter[int] = Counter()
    primitive_counts: Counter[int] = Counter()
    height_values: list[float] = []
    tilt_values: list[float] = []
    basis_ids_present: set[int] = set()

    for frame in frames:
        info = dict(getattr(frame, "info", {}) or {})
        primitive_index = int(info.get("primitive_index", -1))
        primitive_counts[primitive_index] += 1
        if "rotation_index" in info:
            rotation_counts[int(info["rotation_index"])] += 1
        if "azimuth_index" in info:
            azimuth_counts[int(info["azimuth_index"])] += 1
        if "height" in info:
            height_values.append(float(info["height"]))
        basis_id_raw = info.get("basis_id", None)
        if basis_id_raw is not None and int(basis_id_raw) >= 0:
            basis_ids_present.add(int(basis_id_raw))

        site_meta = primitive_meta.get(primitive_index, {})
        site_label = str(
            info.get("site_label")
            or site_meta.get("site_label")
            or info.get("site_kind")
            or site_meta.get("site_kind")
            or f"primitive_{primitive_index}"
        )
        row = site_rows.setdefault(
            site_label,
            {
                "site_label": site_label,
                "primitive_indices": set(),
                "basis_ids": set(),
                "site_kinds": set(),
                "count": 0,
                "rotation_indices": set(),
                "azimuth_indices": set(),
                "tilt_deg_values": [],
                "tilt_bin_counts": Counter(),
                "height_values": [],
            },
        )
        row["primitive_indices"].add(int(primitive_index))
        if basis_id_raw is not None and int(basis_id_raw) >= 0:
            row["basis_ids"].add(int(basis_id_raw))
        site_kind = str(info.get("site_kind") or site_meta.get("site_kind") or "")
        if site_kind:
            row["site_kinds"].add(site_kind)
        if "rotation_index" in info:
            row["rotation_indices"].add(int(info["rotation_index"]))
        if "azimuth_index" in info:
            row["azimuth_indices"].add(int(info["azimuth_index"]))
        if "height" in info:
            row["height_values"].append(float(info["height"]))
        row["count"] += 1

        ads = frame[int(slab_n) :].copy() if int(slab_n) > 0 else frame.copy()
        normal = primitive_normals.get(primitive_index, np.array([0.0, 0.0, 1.0], dtype=float))
        tilt_deg = float(PoseSampler._estimate_tilt_deg(ads, normal))
        tilt_values.append(tilt_deg)
        row["tilt_deg_values"].append(tilt_deg)
        tilt_bin = classify_tilt_bin(tilt_deg)
        row["tilt_bin_counts"][tilt_bin] += 1
        orientation_counts[tilt_bin] += 1

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    rows_out: list[dict[str, Any]] = []
    for site_label in sorted(site_rows.keys()):
        row = site_rows[site_label]
        rows_out.append(
            {
                "site_label": str(site_label),
                "count": int(row["count"]),
                "primitive_indices": [int(v) for v in sorted(row["primitive_indices"])],
                "basis_ids": [int(v) for v in sorted(row["basis_ids"])],
                "site_kinds": [str(v) for v in sorted(row["site_kinds"])],
                "rotation_indices": [int(v) for v in sorted(row["rotation_indices"])],
                "azimuth_indices": [int(v) for v in sorted(row["azimuth_indices"])],
                "tilt_bin_counts": {str(k): int(v) for k, v in sorted(row["tilt_bin_counts"].items())},
                "tilt_deg_stats": _stats(row["tilt_deg_values"]),
                "height_stats": _stats(row["height_values"]),
            }
        )

    return {
        "n_pose_frames": int(len(frames)),
        "n_unique_primitives": int(len({k for k in primitive_counts.keys() if k >= 0})),
        "basis_ids_present": [int(v) for v in sorted(basis_ids_present)],
        "primitive_pose_counts": {str(k): int(v) for k, v in sorted(primitive_counts.items())},
        "rotation_index_counts": {str(k): int(v) for k, v in sorted(rotation_counts.items())},
        "azimuth_index_counts": {str(k): int(v) for k, v in sorted(azimuth_counts.items())},
        "orientation_bin_counts": {str(k): int(v) for k, v in sorted(orientation_counts.items())},
        "tilt_deg_stats": _stats(tilt_values),
        "height_stats": _stats(height_values),
        "sites": rows_out,
    }

