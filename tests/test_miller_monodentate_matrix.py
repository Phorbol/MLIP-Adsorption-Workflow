import json
import subprocess
import sys
from pathlib import Path


def test_miller_monodentate_matrix_smoke():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "run_miller_monodentate_matrix.py"
    out_root = repo / "artifacts" / "autoresearch" / "physics_audit" / "miller_monodentate_matrix_smoke"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--max-slabs",
            "3",
            "--max-molecules",
            "4",
        ],
        check=True,
        cwd=repo,
    )
    summary = out_root / "miller_monodentate_matrix_summary.json"
    assert summary.exists(), "Miller x monodentate summary was not generated."
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload.get("matrix_kind") == "miller_metal_x_monodentate"
    assert int(payload.get("n_slabs", 0)) == 3
    assert int(payload.get("n_molecules", 0)) == 4
    assert int(payload.get("n_pairs", 0)) == 12
    assert float(payload.get("ok_ratio", 0.0)) > 0.0
