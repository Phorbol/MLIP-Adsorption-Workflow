# Progressive Teaching Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a teaching-first notebook that guides users from the simplest workflow usage to the current recommended production MACE workflow and then to difficult heterogeneous-support cases.

**Architecture:** Add one new notebook under `examples/` rather than modifying the existing full-usage notebook. Structure the notebook into staged markdown-plus-code sections so the user can read, run, inspect outputs, and then escalate to more advanced usage without changing files elsewhere in the repo.

**Tech Stack:** Jupyter notebook JSON (`nbformat` 4), Python, ASE, public `adsorption_ensemble` workflow APIs, optional MACE backend.

---

### Task 1: Prepare notebook outline and content mapping

**Files:**
- Create: `examples/progressive_teaching_workflow.ipynb`
- Reference: `docs/superpowers/specs/2026-04-18-progressive-teaching-notebook-design.md`
- Reference: `README.md`
- Reference: `examples/full_usage.ipynb`

- [ ] **Step 1: Confirm the target section sequence from the approved spec**

Sections to include:

```text
0. Title and learning goals
1. Environment check
2. Minimal high-level workflow
3. How to read outputs
4. Recommended production usage
5. Reading production results
6. Escalating to a difficult heterogeneous-support case
7. Real repo case studies
8. Practical tuning guide
```

- [ ] **Step 2: Reuse only current public APIs**

Notebook code should import:

```python
from adsorption_ensemble.relax import IdentityRelaxBackend, MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    DEFAULT_MACE_HEAD_NAME,
    DEFAULT_MACE_MODEL_PATH,
    generate_adsorption_ensemble,
    make_adsorption_workflow_config,
    make_default_surface_preprocessor,
    make_sampling_schedule,
    run_adsorption_workflow,
)
```

- [ ] **Step 3: Define the teaching path**

The notebook should use:

```text
minimal example -> inspect result -> production example -> inspect production counts -> advanced case recipe
```

### Task 2: Build the notebook file

**Files:**
- Create: `examples/progressive_teaching_workflow.ipynb`

- [ ] **Step 1: Add markdown cells for the teaching narrative**

Markdown cells must explain:

```text
- what the workflow is doing
- why the current section exists
- what to inspect after each run
- what downgrade mode means
- why advanced-case overrides are case-scoped only
```

- [ ] **Step 2: Add an environment-check code cell**

The environment-check code cell should perform:

```python
from pathlib import Path
import importlib
import json
import os

status = {}
status["adsorption_ensemble"] = importlib.util.find_spec("adsorption_ensemble") is not None
status["ase"] = importlib.util.find_spec("ase") is not None
status["torch"] = importlib.util.find_spec("torch") is not None
```

And then extend it to check:

```python
import torch
cuda_available = bool(torch.cuda.is_available())
model_path = Path(os.environ.get("AE_MACE_MODEL_PATH", DEFAULT_MACE_MODEL_PATH))
model_exists = model_path.exists()
```

- [ ] **Step 3: Add the minimal workflow code cell**

The minimal workflow code should resemble:

```python
from pathlib import Path
from ase.build import fcc111, molecule

slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
adsorbate = molecule("NH3")

if use_mace_backend:
    quick_backend = MACEBatchRelaxBackend(
        MaceRelaxConfig(
            model_path=str(model_path),
            device="cuda",
            dtype="float32",
            head_name=DEFAULT_MACE_HEAD_NAME,
            enable_cueq=True,
            strict=True,
            max_edges_per_batch=100000,
        )
    )
else:
    quick_backend = IdentityRelaxBackend()

quick_result = generate_adsorption_ensemble(
    slab=slab,
    adsorbate=adsorbate,
    work_dir=Path("artifacts/notebook_progressive/minimal_fcc111_nh3"),
    placement_mode="anchor_free",
    schedule=make_sampling_schedule("multistage_default"),
    dedup_metric="binding_surface_distance",
    signature_mode="provenance",
    basin_relax_backend=quick_backend,
)
```

- [ ] **Step 4: Add a production usage code cell**

The production cell should resemble:

```python
slab_prod = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
adsorbate_prod = molecule("CO")

prod_backend = MACEBatchRelaxBackend(
    MaceRelaxConfig(
        model_path=str(model_path),
        device="cuda",
        dtype="float32",
        head_name=DEFAULT_MACE_HEAD_NAME,
        enable_cueq=True,
        strict=True,
        max_edges_per_batch=100000,
    )
)

prod_result = generate_adsorption_ensemble(
    slab=slab_prod,
    adsorbate=adsorbate_prod,
    work_dir=Path("artifacts/notebook_progressive/production_pt211_co"),
    placement_mode="anchor_free",
    schedule=make_sampling_schedule("multistage_default"),
    dedup_metric="binding_surface_distance",
    signature_mode="provenance",
    basin_overrides={
        "dedup_cluster_method": "greedy",
        "surface_descriptor_threshold": 0.30,
        "surface_descriptor_nearest_k": 8,
        "surface_descriptor_atom_mode": "binding_only",
        "surface_descriptor_relative": False,
        "surface_descriptor_rmsd_gate": 0.25,
        "desorption_min_bonds": 1,
        "energy_window_ev": 2.5,
    },
    basin_relax_backend=prod_backend,
)
```

- [ ] **Step 5: Add an advanced-case recipe cell**

The advanced-case cell should demonstrate:

```python
schedule = make_sampling_schedule("multistage_default")
schedule.pre_relax_selection.max_candidates = 64

cfg = make_adsorption_workflow_config(
    work_dir=Path("artifacts/notebook_progressive/advanced_case/workflow"),
    placement_mode="anchor_free",
    single_atom=False,
    exhaustive_pose_sampling=True,
    dedup_metric="binding_surface_distance",
    signature_mode="provenance",
    pose_overrides={
        "adaptive_height_fallback": True,
        "adaptive_height_fallback_step": 0.20,
        "adaptive_height_fallback_max_extra": 1.60,
        "adaptive_height_fallback_contact_slack": 0.60,
    },
    basin_overrides={
        "dedup_cluster_method": "greedy",
        "surface_descriptor_threshold": 0.30,
        "surface_descriptor_nearest_k": 8,
        "surface_descriptor_atom_mode": "binding_only",
        "surface_descriptor_relative": False,
        "surface_descriptor_rmsd_gate": 0.25,
        "desorption_min_bonds": 1,
        "surface_reconstruction_enabled": False,
        "energy_window_ev": 2.5,
    },
)
cfg.surface_preprocessor = make_default_surface_preprocessor(
    target_count_mode="off",
    target_surface_fraction=0.25,
)
cfg.pre_relax_selection = schedule.pre_relax_selection
cfg.basin_config.post_relax_selection = schedule.post_relax_selection
cfg.max_selected_primitives = None
```

### Task 3: Add guardrails and case-study references

**Files:**
- Create: `examples/progressive_teaching_workflow.ipynb`

- [ ] **Step 1: Guard production-only cells**

Production-only cells should check:

```python
if not use_mace_backend:
    print("Skipping this production cell because CUDA/MACE/model availability was not detected.")
else:
    ...
```

- [ ] **Step 2: Add case-study markdown**

Include current audited counts:

```text
Pt(211)+Ag4+C6H6: 352 poses -> 64 selected -> 37 basins
Pt(211)+Ag4+CO:   414 poses -> 64 selected -> 36 basins
```

- [ ] **Step 3: Add a practical tuning markdown checklist**

Checklist items must cover:

```text
- increase pre-relax budget when coverage collapses upstream
- use exhaustive pose sampling for difficult heterogeneous supports
- inspect raw_site_dictionary.json when site coverage is suspicious
- pre-relax the bare support before adsorption on soft cluster-on-slab systems
- keep difficult-case overrides case-scoped
```

### Task 4: Verify notebook structure

**Files:**
- Verify: `examples/progressive_teaching_workflow.ipynb`

- [ ] **Step 1: Parse-check the notebook**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("examples/progressive_teaching_workflow.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
assert nb["nbformat"] == 4
assert "cells" in nb and len(nb["cells"]) >= 12
print("cells", len(nb["cells"]))
PY
```

Expected:

```text
Notebook JSON parses and contains the expected sectioned cell set.
```

- [ ] **Step 2: Verify key imports used by the notebook**

Run:

```bash
python - <<'PY'
from adsorption_ensemble.relax import IdentityRelaxBackend, MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    DEFAULT_MACE_HEAD_NAME,
    DEFAULT_MACE_MODEL_PATH,
    generate_adsorption_ensemble,
    make_adsorption_workflow_config,
    make_default_surface_preprocessor,
    make_sampling_schedule,
    run_adsorption_workflow,
)
print("imports ok")
PY
```

Expected:

```text
imports ok
```
