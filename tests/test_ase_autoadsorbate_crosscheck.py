import json
import subprocess
import sys
from pathlib import Path


def test_ase_autoadsorbate_crosscheck_smoke():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "run_ase_autoadsorbate_crosscheck.py"
    out_root = repo / "artifacts" / "autoresearch" / "physics_audit" / "ase_full_matrix_smoke"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--max-slabs",
            "2",
            "--max-molecules",
            "5",
            "--skip-autoadsorbate",
        ],
        check=True,
        cwd=repo,
    )
    summary = out_root / "ase_autoadsorbate_crosscheck_summary.json"
    assert summary.exists(), "Crosscheck summary was not generated."
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert int(payload.get("n_slabs", 0)) == 2
    assert int(payload.get("n_molecules", 0)) == 5
    assert int(payload.get("n_pairs", 0)) == 10
