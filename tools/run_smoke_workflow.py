from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adsorption_ensemble.workflows.smoke import PoseSamplingSmokeConfig, run_pose_sampling_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="artifacts/smoke")
    parser.add_argument("--mace-disabled", action="store_true")
    parser.add_argument("--mace-enabled", action="store_true")
    args = parser.parse_args()
    mace_disabled = True
    if bool(args.mace_enabled):
        mace_disabled = False
    if bool(args.mace_disabled):
        mace_disabled = True
    smoke_cfg = PoseSamplingSmokeConfig(mace_disabled=bool(mace_disabled))
    out = run_pose_sampling_smoke(out_root=Path(args.out_root), smoke=smoke_cfg)
    print(json.dumps({"run_dir": out["run_dir"], "summary_json": out["summary_json"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
