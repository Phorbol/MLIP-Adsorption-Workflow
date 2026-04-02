import json
import subprocess
import sys
from pathlib import Path


def test_simple_case_validation_smoke_fake_relax():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "run_simple_case_validation.py"
    out_root = repo / "artifacts" / "autoresearch" / "simple_case_validation_smoke_default"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--max-cases",
            "2",
            "--relax-backend",
            "fake",
        ],
        check=True,
        cwd=repo,
    )
    summary = out_root / "simple_case_validation.json"
    assert summary.exists(), "Simple-case validation summary was not generated."
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload.get("relax_backend") == "fake"
    rows = payload.get("rows", [])
    assert len(rows) == 2
    for row in rows:
        assert float(row["overlap"]["manual_recall_by_ours"]) >= 1.0
