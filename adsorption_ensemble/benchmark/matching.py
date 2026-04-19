from __future__ import annotations

from typing import Any


def _candidate_rank(candidate: dict[str, Any]) -> tuple[int, float, int, int]:
    return (
        0 if bool(candidate.get("signature_match", False)) else 1,
        float(candidate.get("rmsd", 1.0e9)),
        int(candidate.get("manual_index", -1)),
        int(candidate.get("ours_index", -1)),
    )


def select_unique_reference_matches(
    candidates: list[dict[str, Any]],
    *,
    n_manual: int,
    n_ours: int,
) -> list[dict[str, Any]]:
    if n_manual <= 0 or n_ours <= 0 or not candidates:
        return []

    by_manual: dict[int, list[dict[str, Any]]] = {int(i): [] for i in range(int(n_manual))}
    for row in candidates:
        m = int(row.get("manual_index", -1))
        o = int(row.get("ours_index", -1))
        if m < 0 or m >= int(n_manual) or o < 0 or o >= int(n_ours):
            continue
        by_manual[m].append(dict(row))
    for rows in by_manual.values():
        rows.sort(key=_candidate_rank)

    manual_order = sorted(
        range(int(n_manual)),
        key=lambda m: (
            len(by_manual.get(m, [])) if by_manual.get(m) else 10**9,
            _candidate_rank(by_manual[m][0]) if by_manual.get(m) else (10**9, 10**9, m, 10**9),
            int(m),
        ),
    )

    owner_of_ours: dict[int, int] = {}
    chosen_by_ours: dict[int, dict[str, Any]] = {}

    def _augment(manual_index: int, seen_ours: set[int]) -> bool:
        for cand in by_manual.get(int(manual_index), []):
            ours_index = int(cand["ours_index"])
            if ours_index in seen_ours:
                continue
            seen_ours.add(ours_index)
            current_owner = owner_of_ours.get(ours_index)
            if current_owner is None or _augment(current_owner, seen_ours):
                owner_of_ours[ours_index] = int(manual_index)
                chosen_by_ours[ours_index] = dict(cand)
                return True
        return False

    for manual_index in manual_order:
        _augment(int(manual_index), set())

    out = sorted(chosen_by_ours.values(), key=_candidate_rank)
    return [dict(row) for row in out]
