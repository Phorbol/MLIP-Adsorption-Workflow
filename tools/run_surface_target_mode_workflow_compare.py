from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.build import molecule

from adsorption_ensemble.relax.backends import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import make_adsorption_workflow_config, make_default_surface_preprocessor, run_adsorption_workflow
from tools.run_ase_autoadsorbate_crosscheck import build_slab_suite


DEFAULT_CASES = (
    ("Fe_bcc111", "CO"),
    ("Pt_fcc211", "CO"),
    ("Ru_hcp10m10", "CO"),
)
DEFAULT_MODES = ("fixed", "off", "adaptive")


def _make_adsorbate(name: str) -> Atoms:
    key = str(name).strip()
    if key == "CO":
        co = molecule("CO")
        if co[0].symbol != "C":
            co = co[[1, 0]]
        return co
    if key == "NH3":
        return molecule("NH3")
    if key == "H2O":
        return molecule("H2O")
    return molecule(key)


def _infer_head_name(model_path: str | None) -> str | None:
    if not model_path:
        return None
    name = Path(model_path).name.lower()
    if "omat" in name:
        return "omat_pbe"
    if "omol" in name:
        return "omol"
    return None


def _parse_cases(raw_cases: list[str] | None) -> list[tuple[str, str]]:
    if not raw_cases:
        return list(DEFAULT_CASES)
    out: list[tuple[str, str]] = []
    for item in raw_cases:
        text = str(item).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Case '{text}' must be formatted as slab:adsorbate.")
        slab_name, ads_name = text.split(":", 1)
        out.append((slab_name.strip(), ads_name.strip()))
    return out


def run_case(
    *,
    slab_name: str,
    ads_name: str,
    slab: Atoms,
    adsorbate: Atoms,
    mode: str,
    out_root: Path,
    model_path: str,
    device: str,
    dtype: str,
) -> dict:
    case_dir = out_root / slab_name / ads_name / mode
    cfg = make_adsorption_workflow_config(case_dir)
    cfg.surface_preprocessor = make_default_surface_preprocessor(
        target_count_mode=str(mode),
        target_surface_fraction=0.25,
    )
    relax_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=str(model_path),
            device=str(device),
            dtype=str(dtype),
            head_name=_infer_head_name(str(model_path)),
            enable_cueq=True,
            strict=True,
            max_edges_per_batch=20000,
        )
    )
    result = run_adsorption_workflow(
        slab=slab,
        adsorbate=adsorbate,
        config=cfg,
        basin_relax_backend=relax_backend,
    )
    summary = dict(result.summary)
    summary["surface_target_mode"] = str(mode)
    summary["work_dir"] = case_dir.as_posix()
    return summary


def _write_markdown(out_path: Path, payload: dict) -> None:
    lines = ["# Surface Target Mode Workflow Compare", ""]
    lines.append(f"- model_path: `{payload['model_path']}`")
    lines.append(f"- device: `{payload['device']}`")
    lines.append(f"- dtype: `{payload['dtype']}`")
    lines.append("")
    lines.append("| slab | adsorbate | mode | surface_atoms | basis_primitives | pose_frames | basins | nodes | decision |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload["rows"]:
        lines.append(
            "| {slab} | {adsorbate} | {mode} | {surface_atoms} | {basis} | {poses} | {basins} | {nodes} | {decision} |".format(
                slab=row["slab"],
                adsorbate=row["adsorbate"],
                mode=row["mode"],
                surface_atoms=row["n_surface_atoms"],
                basis=row["n_basis_primitives"],
                poses=row["n_pose_frames"],
                basins=row["n_basins"],
                nodes=row["n_nodes"],
                decision=row.get("adaptive_target_decision"),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=str,
        default="artifacts/autoresearch/surface_target_mode_workflow_compare_20260405",
    )
    parser.add_argument("--mace-model-path", type=str, default="/root/.cache/mace/mace-omat-0-small.model")
    parser.add_argument("--mace-device", type=str, default="cuda")
    parser.add_argument("--mace-dtype", type=str, default="float32")
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    slabs = build_slab_suite()
    cases = _parse_cases(args.cases)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for slab_name, ads_name in cases:
        if slab_name not in slabs:
            raise ValueError(f"Unknown slab case: {slab_name}")
        slab = slabs[slab_name]
        adsorbate = _make_adsorbate(ads_name)
        for mode in args.modes:
            summary = run_case(
                slab_name=slab_name,
                ads_name=ads_name,
                slab=slab,
                adsorbate=adsorbate,
                mode=str(mode),
                out_root=out_root,
                model_path=str(args.mace_model_path),
                device=str(args.mace_device),
                dtype=str(args.mace_dtype),
            )
            rows.append(
                {
                    "slab": slab_name,
                    "adsorbate": ads_name,
                    "mode": str(mode),
                    "n_surface_atoms": int(summary["n_surface_atoms"]),
                    "n_basis_primitives": int(summary["n_basis_primitives"]),
                    "n_pose_frames": int(summary["n_pose_frames"]),
                    "n_basins": int(summary["n_basins"]),
                    "n_nodes": int(summary["n_nodes"]),
                    "adaptive_target_decision": summary.get("surface_diagnostics", {}).get("adaptive_target_decision"),
                    "work_dir": str(summary["work_dir"]),
                }
            )

    payload = {
        "model_path": str(args.mace_model_path),
        "device": str(args.mace_device),
        "dtype": str(args.mace_dtype),
        "rows": rows,
    }
    out_json = out_root / "surface_target_mode_workflow_compare.json"
    out_md = out_root / "surface_target_mode_workflow_compare.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    print(out_json.as_posix())
    print(out_md.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
