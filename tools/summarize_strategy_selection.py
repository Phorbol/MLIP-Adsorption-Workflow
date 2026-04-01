from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    path = Path(args.summary_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    ranking = data.get("ranking", [])
    rows = data.get("rows", [])
    print(f"total_case_runs: {data.get('n_case_runs', len(rows))}")
    print("top_strategies:")
    for r in ranking[: max(1, int(args.top_k))]:
        print(
            json.dumps(
                {
                    "strategy": r.get("strategy"),
                    "energy_window_ev": r.get("energy_window_ev"),
                    "mace_node_l2_threshold": r.get("mace_node_l2_threshold"),
                    "mean_score": round(float(r.get("mean_score", 0.0)), 4),
                    "nonzero_case_rate": round(float(r.get("nonzero_case_rate", 0.0)), 4),
                    "mean_runtime_sec": round(float(r.get("mean_runtime_sec", 0.0)), 3),
                    "mean_basins": round(float(r.get("mean_basins", 0.0)), 3),
                },
                ensure_ascii=False,
            )
        )
    # strategy-level aggregates
    by_strategy: dict[str, list[dict]] = {}
    for r in ranking:
        by_strategy.setdefault(str(r.get("strategy")), []).append(r)
    print("strategy_aggregates:")
    for s, items in sorted(by_strategy.items()):
        best = max(items, key=lambda x: float(x.get("mean_score", 0.0)))
        mean_score = sum(float(x.get("mean_score", 0.0)) for x in items) / len(items)
        mean_nonzero = sum(float(x.get("nonzero_case_rate", 0.0)) for x in items) / len(items)
        print(
            json.dumps(
                {
                    "strategy": s,
                    "best_mean_score": round(float(best.get("mean_score", 0.0)), 4),
                    "best_energy_window_ev": best.get("energy_window_ev"),
                    "best_mace_l2_threshold": best.get("mace_node_l2_threshold"),
                    "avg_mean_score": round(mean_score, 4),
                    "avg_nonzero_case_rate": round(mean_nonzero, 4),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

