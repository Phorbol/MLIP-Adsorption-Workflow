from __future__ import annotations

import argparse
import json
from pathlib import Path

from adsorption_ensemble.benchmark import audit_cu111_co_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=str,
        default="artifacts/autoresearch/final_basin_validation/miller_monodentate_surfacefix_macefinal_smoke_v3/Cu_fcc111/CO",
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    payload = audit_cu111_co_case(args.case_dir)
    out_path = Path(args.output) if str(args.output).strip() else Path(args.case_dir) / "cu111_co_sentinel_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
