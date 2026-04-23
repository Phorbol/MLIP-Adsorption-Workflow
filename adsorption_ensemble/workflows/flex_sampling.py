from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def _build_bond_graph(atoms: Atoms, tau: float = 1.20) -> list[set[int]]:
    n = len(atoms)
    adj = [set() for _ in range(n)]
    if n <= 1:
        return adj
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    d = atoms.get_all_distances(mic=False)
    for i in range(n):
        ri = float(covalent_radii[int(z[i])])
        for j in range(i + 1, n):
            rj = float(covalent_radii[int(z[j])])
            if float(d[i, j]) <= float(tau) * (ri + rj):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _n_components(adj: list[set[int]]) -> int:
    n = len(adj)
    seen = [False] * n
    comps = 0
    for i in range(n):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
    return comps


@dataclass
class FlexSamplingBudget:
    run_conformer_search: bool
    md_time_ps: float
    md_runs: int
    preselect_k: int
    target_final_k: int
    selection_profile: str
    fps_rounds: int
    fps_round_size: int
    score: float
    rationale: dict[str, float | int | bool | str]


def plan_flex_sampling_budget(
    adsorbate: Atoms,
    *,
    n_surface_atoms: int | None = None,
    n_site_primitives: int | None = None,
) -> FlexSamplingBudget:
    z = np.asarray(adsorbate.get_atomic_numbers(), dtype=int)
    heavy = [i for i, zi in enumerate(z.tolist()) if int(zi) > 1]
    n_heavy = int(len(heavy))
    adj = _build_bond_graph(adsorbate)
    heavy_edges = 0
    rotatable_proxy = 0
    for i in heavy:
        for j in adj[i]:
            if j <= i or j not in heavy:
                continue
            heavy_edges += 1
            if len(adj[i]) >= 2 and len(adj[j]) >= 2:
                rotatable_proxy += 1
    comps = _n_components([set([j for j in nbrs if j in heavy]) for nbrs in adj]) if heavy else 0
    ring_proxy = max(0, int(heavy_edges - n_heavy + comps))
    hetero = sum(1 for zi in z.tolist() if int(zi) not in {1, 6})
    pos = np.asarray(adsorbate.get_positions(), dtype=float)
    span = 0.0 if len(pos) == 0 else float(np.max(np.ptp(pos, axis=0)))
    surf_complexity = 0.0
    if n_surface_atoms is not None:
        surf_complexity += float(max(0.0, (float(n_surface_atoms) - 16.0) / 24.0))
    if n_site_primitives is not None:
        surf_complexity += float(max(0.0, (float(n_site_primitives) - 24.0) / 48.0))
    score = (
        0.35 * float(rotatable_proxy)
        + 0.80 * float(ring_proxy)
        + 0.18 * float(max(0, n_heavy - 4))
        + 0.22 * float(hetero)
        + 0.25 * float(max(0.0, span - 3.0))
        + 0.30 * surf_complexity
    )
    run_conformer_search = bool((n_heavy >= 7 and score >= 2.0) or score >= 3.2 or rotatable_proxy >= 2)
    if not run_conformer_search:
        return FlexSamplingBudget(
            run_conformer_search=False,
            md_time_ps=0.0,
            md_runs=0,
            preselect_k=0,
            target_final_k=0,
            selection_profile="disabled",
            fps_rounds=0,
            fps_round_size=0,
            score=float(score),
            rationale={
                "n_heavy": int(n_heavy),
                "rotatable_proxy": int(rotatable_proxy),
                "ring_proxy": int(ring_proxy),
                "hetero_count": int(hetero),
                "span": float(span),
                "surface_complexity": float(surf_complexity),
                "score": float(score),
                "run_conformer_search": False,
                "target_final_k": 0,
                "selection_profile": "disabled",
            },
        )
    md_time_ps = float(min(40.0, max(8.0, 8.0 + 2.2 * score)))
    md_runs = int(min(6, max(2, round(2 + score / 2.2))))
    preselect_k = int(min(256, max(64, round(64 + 18 * score))))
    target_final_k = int(12 if score >= 5.0 else 8)
    fps_rounds = int(min(16, max(4, round(4 + score / 1.4))))
    fps_round_size = int(min(48, max(12, round(12 + 4 * score))))
    return FlexSamplingBudget(
        run_conformer_search=True,
        md_time_ps=md_time_ps,
        md_runs=md_runs,
        preselect_k=preselect_k,
        target_final_k=target_final_k,
        selection_profile="adsorption_seed_broad",
        fps_rounds=fps_rounds,
        fps_round_size=fps_round_size,
        score=float(score),
        rationale={
            "n_heavy": int(n_heavy),
            "rotatable_proxy": int(rotatable_proxy),
            "ring_proxy": int(ring_proxy),
            "hetero_count": int(hetero),
            "span": float(span),
            "surface_complexity": float(surf_complexity),
            "score": float(score),
            "run_conformer_search": True,
            "md_time_ps": float(md_time_ps),
            "md_runs": int(md_runs),
            "preselect_k": int(preselect_k),
            "target_final_k": int(target_final_k),
            "selection_profile": "adsorption_seed_broad",
            "fps_rounds": int(fps_rounds),
            "fps_round_size": int(fps_round_size),
        },
    )
