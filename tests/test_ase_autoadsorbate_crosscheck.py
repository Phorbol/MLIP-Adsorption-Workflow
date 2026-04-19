import json
import os
import subprocess
import sys
from pathlib import Path


def test_ase_autoadsorbate_crosscheck_smoke():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "run_ase_autoadsorbate_crosscheck.py"
    out_root = repo / "artifacts" / "autoresearch" / "physics_audit" / "ase_reference_triplet_smoke"
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(repo / "artifacts" / "autoresearch" / "physics_audit" / ".mplconfig"))
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--slab-names",
            "Pt_fcc111",
            "Fe_bcc110",
            "Ru_hcp0001",
            "--molecule-names",
            "H",
            "CO",
            "NH3",
            "--skip-autoadsorbate",
            "--no-save-sample-placements",
            "--no-save-visuals",
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    summary = out_root / "ase_autoadsorbate_crosscheck_summary.json"
    assert summary.exists(), "Crosscheck summary was not generated."
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload.get("n_slabs") == 3
    assert payload.get("n_molecules") == 3
    assert payload.get("n_pairs") == 9
    assert payload.get("placement_stats", {}).get("succeeded") == 9
    assert payload.get("failed_pairs") == []

    rows = {str(row["slab"]): row for row in payload.get("slab_rows", [])}
    assert set(rows) == {"Pt_fcc111", "Fe_bcc110", "Ru_hcp0001"}
    assert rows["Pt_fcc111"]["ase_reference"]["observed_mapped_counts"] == {"ontop": 1, "bridge": 1, "hollow": 2}
    assert rows["Fe_bcc110"]["ase_reference"]["observed_mapped_counts"] == {"ontop": 1, "bridge": 2, "hollow": 1}
    assert rows["Ru_hcp0001"]["ase_reference"]["observed_mapped_counts"] == {"ontop": 1, "bridge": 1, "hollow": 2}
    assert all(bool(row["ase_reference"].get("strict_mapped_counts_match")) for row in rows.values())
