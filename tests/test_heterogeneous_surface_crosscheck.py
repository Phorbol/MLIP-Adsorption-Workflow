import json
import os
import subprocess
import sys
from pathlib import Path


def test_heterogeneous_surface_crosscheck_smoke():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "run_ase_autoadsorbate_crosscheck.py"
    out_root = repo / "artifacts" / "autoresearch" / "physics_audit" / "heterogeneous_surface_reference_smoke"
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(repo / "artifacts" / "autoresearch" / "physics_audit" / ".mplconfig"))
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--slab-names",
            "CuNi_fcc111_alloy",
            "TiO2_110",
            "MgO_100",
            "Pt_fcc111_vacancy",
            "Pt_fcc111_adatom",
            "Pt_fcc111_cluster_interface",
            "--molecule-names",
            "H",
            "CO",
            "NH3",
            "H2O",
            "--skip-autoadsorbate",
            "--no-save-sample-placements",
            "--no-save-visuals",
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    summary = out_root / "ase_autoadsorbate_crosscheck_summary.json"
    assert summary.exists(), "Heterogeneous surface crosscheck summary was not generated."
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload.get("n_slabs") == 6
    assert payload.get("n_molecules") == 4
    assert payload.get("n_pairs") == 24
    assert payload.get("placement_stats", {}).get("succeeded") == 24
    assert payload.get("failed_pairs") == []

    rows = {str(row["slab"]): row for row in payload.get("slab_rows", [])}
    assert rows["CuNi_fcc111_alloy"]["ours"]["n_basis_primitives"] >= 10
    assert rows["CuNi_fcc111_alloy"]["ours"]["basis_counts"].get("1c", 0) >= 2
    assert rows["CuNi_fcc111_alloy"]["ours"]["basis_counts"].get("2c", 0) >= 3
    assert rows["CuNi_fcc111_alloy"]["ours"]["basis_counts"].get("3c", 0) >= 6
    assert rows["TiO2_110"]["ours"]["n_surface_atoms"] == 24
    assert rows["TiO2_110"]["ours"]["n_basis_primitives"] >= 16
    assert rows["TiO2_110"]["ours"]["basis_counts"].get("1c", 0) >= 5
    assert rows["TiO2_110"]["ours"]["basis_counts"].get("2c", 0) >= 7
    assert rows["TiO2_110"]["ours"]["basis_counts"].get("3c", 0) >= 4
    assert rows["MgO_100"]["ours"]["n_surface_atoms"] == 8
    assert rows["MgO_100"]["ours"]["n_basis_primitives"] >= 6
    assert rows["Pt_fcc111_vacancy"]["ours"]["n_basis_primitives"] >= 10
    assert rows["Pt_fcc111_adatom"]["ours"]["n_basis_primitives"] >= 10
    assert rows["Pt_fcc111_cluster_interface"]["ours"]["n_basis_primitives"] >= 15
