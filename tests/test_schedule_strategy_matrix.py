import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestScheduleStrategyMatrix(unittest.TestCase):
    def test_schedule_strategy_matrix_smoke(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo / "tools" / "run_schedule_strategy_matrix.py"
        out_root = repo / "artifacts" / "autoresearch" / "schedule_strategy_matrix_smoke"
        subprocess.run([sys.executable, str(script), "--out-root", str(out_root)], check=True, cwd=repo)
        out = out_root / "schedule_strategy_matrix.json"
        self.assertTrue(out.exists())
        payload = json.loads(out.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        self.assertEqual(len(rows), 40)
        self.assertTrue(any(int(r["n_pose_frames_selected_for_basin"]) < int(r["n_pose_frames"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
