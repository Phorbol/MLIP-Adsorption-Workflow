# Pt211 Ag4 Production Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `Pt(211)+Ag4+C6H6` run behave like a production case by removing the primitive cap for this case, allowing a case-scoped surface-reconstruction toggle, and adding a case-scoped adaptive-height fallback so Ag-related sites have a better chance to produce non-colliding initial poses.

**Architecture:** Keep repo defaults unchanged. Add new config switches with conservative defaults so existing behavior is stable, then enable them only in the case driver under `runs/20260417-234454-pt211-ag4-c6h6/outputs/run_pt211_ag4_c6h6_case.py`. Verify with focused unit tests plus a fresh GPU-backed rerun of the production case.

**Tech Stack:** Python, ASE, pytest, existing adsorption workflow, MACE relax backend

---

### Task 1: Case-Scoped Anomaly Toggle

**Files:**
- Modify: `adsorption_ensemble/basin/types.py`
- Modify: `adsorption_ensemble/basin/anomaly.py`
- Test: `tests/test_binding_pairs.py`

- [ ] **Step 1: Write the failing test**

Add a test in `tests/test_binding_pairs.py` that constructs a relaxed frame with large slab displacement and verifies `classify_anomaly(...)` does **not** return `surface_reconstruction` when a new boolean switch is disabled.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_binding_pairs.py -q`
Expected: FAIL because the new toggle argument does not exist yet or `surface_reconstruction` is still returned.

- [ ] **Step 3: Write minimal implementation**

Add `surface_reconstruction_enabled: bool = True` to `BasinConfig` and gate the displacement check in `classify_anomaly(...)` behind that boolean while preserving the old default behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_binding_pairs.py -q`
Expected: PASS

### Task 2: Adaptive-Height Fallback For Zero-Pose Primitives

**Files:**
- Modify: `adsorption_ensemble/pose/sampler.py`
- Test: `tests/test_pose_sampler.py`

- [ ] **Step 1: Write the failing test**

Add a focused test in `tests/test_pose_sampler.py` for a difficult surface/primitive setup where the default settings produce zero poses, then enable a new adaptive fallback switch and assert that the same primitive yields at least one pose.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pose_sampler.py -q`
Expected: FAIL because the new fallback switch does not exist or behavior is unchanged.

- [ ] **Step 3: Write minimal implementation**

Add case-scoped config fields to `PoseSamplerConfig`, for example:
- `adaptive_height_fallback: bool = False`
- `adaptive_height_fallback_max_extra: float = 1.5`
- `adaptive_height_fallback_step: float = 0.2`

In `PoseSampler.sample(...)`, if a primitive produces zero candidates under the normal height solver, retry that primitive with a higher `max_height` window and a looser incremental search along the site normal. Keep the fallback local to that primitive and only when the switch is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pose_sampler.py -q`
Expected: PASS

### Task 3: Production Case Driver Upgrade

**Files:**
- Modify: `runs/20260417-234454-pt211-ag4-c6h6/outputs/run_pt211_ag4_c6h6_case.py`

- [ ] **Step 1: Update case-only workflow configuration**

Change the case driver to:
- set `cfg.max_selected_primitives = None`
- use `exhaustive_pose_sampling=True`
- raise pre-relax FPS budget from `24` to `64`
- set `surface_reconstruction_enabled=False` in `basin_overrides`
- enable the adaptive-height fallback in `pose_overrides`

- [ ] **Step 2: Preserve current production safeguards**

Keep:
- bare-support pre-relaxation
- MACE strict GPU configuration
- current dedup metric and post-relax selection

- [ ] **Step 3: Static validation**

Run: `python -m py_compile runs/20260417-234454-pt211-ag4-c6h6/outputs/run_pt211_ag4_c6h6_case.py`
Expected: no output, exit 0

### Task 4: Verification And Production Rerun

**Files:**
- Reuse existing case output directory under `artifacts/autoresearch/single_cases/pt211_ag4_cluster_c6h6_20260417/`
- Update ledger files under `runs/20260417-234454-pt211-ag4-c6h6/`

- [ ] **Step 1: Run focused verification**

Run: `pytest -q tests/test_binding_pairs.py tests/test_pose_sampler.py tests/test_basin_reporting.py tests/test_workflow_api.py tests/test_adsorption_workflow.py`
Expected: PASS

- [ ] **Step 2: Run the formal production case**

Run in `mace_les` with GPU:
`python runs/20260417-234454-pt211-ag4-c6h6/outputs/run_pt211_ag4_c6h6_case.py`

- [ ] **Step 3: Audit Ag-site coverage**

Record:
- number of Ag-related raw primitives
- number of Ag-related selected primitives
- number of Ag-related poses in `pose_pool`
- number of Ag-related selected poses in `pose_pool_selected`
- whether final basins bind to `Ag`, `Pt`, or both

- [ ] **Step 4: Update ledger summary**

Register new artifacts and write the final evidence-backed conclusion to:
- `runs/20260417-234454-pt211-ag4-c6h6/artifacts.json`
- `runs/20260417-234454-pt211-ag4-c6h6/summary.md`
