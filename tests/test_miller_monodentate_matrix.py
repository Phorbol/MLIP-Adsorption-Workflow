import json
import os
import subprocess
import sys
from pathlib import Path


def test_miller_monodentate_matrix_smoke():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "run_miller_monodentate_matrix.py"
    out_root = repo / "artifacts" / "autoresearch" / "physics_audit" / "miller_monodentate_reference_smoke"
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(repo / "artifacts" / "autoresearch" / "physics_audit" / ".mplconfig"))
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--slab-names",
            "Pt_fcc100",
            "Pt_fcc111",
            "Cu_fcc111",
            "Fe_bcc110",
            "Ru_hcp0001",
            "--molecule-names",
            "H",
            "CO",
            "NH3",
            "--no-save-visuals",
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    summary = out_root / "miller_monodentate_matrix_summary.json"
    assert summary.exists(), "Miller x monodentate summary was not generated."
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload.get("matrix_kind") == "miller_metal_x_monodentate"
    assert payload.get("n_slabs") == 5
    assert payload.get("n_molecules") == 3
    assert payload.get("n_pairs") == 15
    assert float(payload.get("ok_ratio", 0.0)) == 1.0
    assert payload.get("failed_pairs") == []

    rows = {str(row["slab"]): row for row in payload.get("slab_rows", [])}
    assert set(rows) == {"Pt_fcc100", "Pt_fcc111", "Cu_fcc111", "Fe_bcc110", "Ru_hcp0001"}
    assert rows["Pt_fcc100"]["basis_counts"] == {"1c": 1, "2c": 1, "4c": 1}
    assert rows["Pt_fcc111"]["basis_counts"] == {"1c": 1, "2c": 1, "3c": 2}
    assert rows["Cu_fcc111"]["basis_counts"] == {"1c": 1, "2c": 1, "3c": 2}
    assert rows["Fe_bcc110"]["basis_counts"] == {"1c": 1, "2c": 2, "3c": 1}
    assert rows["Ru_hcp0001"]["basis_counts"] == {"1c": 1, "2c": 1, "3c": 2}
    assert all(bool(row["ase_reference"].get("strict_mapped_counts_match")) for row in rows.values())
