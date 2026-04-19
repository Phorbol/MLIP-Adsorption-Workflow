# Progressive Teaching Notebook Design

## Goal

Create a new teaching-first Jupyter notebook for this repo that serves both:

- new users who need to understand the workflow from the high-level API upward
- production users who want the currently recommended MACE-backed usage pattern

The notebook should be explanatory rather than minimal. It should teach the user how to think about the workflow, not just provide code cells.

## User Intent

The requested notebook should:

- be a gradual tutorial rather than a bare example
- cover both beginner and production use cases
- default to the production environment and recommended MACE configuration
- include downgrade notes for users without GPU or without a working MACE installation

The requested tone is teaching-oriented:

- more narrative explanation
- shorter, staged code cells
- explicit "what to look at next" guidance after each step

## Target File

- `examples/progressive_teaching_workflow.ipynb`

This notebook should live alongside the existing:

- `examples/full_usage.ipynb`
- `examples/FULL_USAGE.md`

The new notebook is not a replacement for `full_usage.ipynb`. It is a guided tutorial version with clearer staging and more explanation.

## Notebook Positioning

### Existing material

Current repo materials already cover:

- broad README-level usage guidance
- a full-feature example notebook
- low-level scripts in `tools/`
- run-local production case drivers under `runs/`

What is missing is a single, user-facing, progressive notebook that explains:

1. what the workflow stages mean
2. which public entry points users should choose
3. how to move from smoke usage to production usage
4. how to read the generated artifacts
5. when and how to escalate to a harder heterogeneous-support recipe

### Intended role of the new notebook

The new notebook should become the best first notebook for a user opening the repo for the first time.

It should answer:

- "What is this pipeline doing?"
- "What function should I call first?"
- "How do I run it in production with MACE?"
- "What files should I inspect?"
- "What changes for a hard case?"

## Recommended Structure

The notebook should be a single file with strong internal sectioning.

### Part 0. Title and learning goals

Purpose:

- orient the reader
- define the three levels of usage covered in the notebook

Content:

- a short intro paragraph
- a bulleted learning roadmap
- a compact workflow chain:
  - `slab -> surface/site -> pose -> basin -> node`

### Part 1. Environment check

Purpose:

- verify whether the reader is in the intended production setup
- provide immediate downgrade guidance

Content:

- import checks for `adsorption_ensemble`, `ase`, and optional MACE dependencies
- environment detection for CUDA availability
- model path discovery using `AE_MACE_MODEL_PATH` and the repo's current default path
- a small status table or printed summary

Downgrade guidance:

- if GPU or MACE is missing, explain that the tutorial can still run the conceptual path with `IdentityRelaxBackend`
- clearly mark that such a run is for workflow understanding, not physical conclusions

### Part 2. The minimal high-level workflow

Purpose:

- teach the simplest public entry point
- let the user see a full result object before worrying about production details

Content:

- build a simple slab such as `fcc111("Pt", ...)`
- build a simple adsorbate such as `molecule("NH3")` or `molecule("CO")`
- call `generate_adsorption_ensemble(...)`
- default to production path if MACE is available, otherwise downgrade to `IdentityRelaxBackend`

Teaching focus:

- explain `work_dir`
- explain `placement_mode`
- explain `schedule`
- explain what the returned `summary` and `files` mean

### Part 3. How to read the outputs

Purpose:

- prevent the notebook from being just "run code and move on"

Content:

- inspect `result.summary`
- inspect a few high-value file paths from `result.files`
- explain:
  - `site_dictionary.json`
  - `pose_pool.extxyz`
  - `pose_pool_selected.extxyz`
  - `basins.json`
  - `nodes.json`

Teaching focus:

- what each artifact answers
- what a healthy workflow output looks like

### Part 4. Recommended production usage

Purpose:

- show the mainline configuration users should copy for serious runs

Content:

- explicit `MACEBatchRelaxBackend`
- `MaceRelaxConfig` with:
  - `device="cuda"`
  - `dtype="float32"`
  - `head_name=DEFAULT_MACE_HEAD_NAME`
  - `enable_cueq=True`
  - `strict=True`
  - `max_edges_per_batch=100000`
- `generate_adsorption_ensemble(...)`
- basin overrides matching the current README recommendation

Teaching focus:

- why this is the recommended entry point for most users
- what each important production parameter is doing
- what not to tweak first

### Part 5. Reading production results

Purpose:

- help the user evaluate whether a run is useful

Content:

- show the counts:
  - `n_surface_atoms`
  - `n_basis_primitives`
  - `n_pose_frames`
  - `n_pose_frames_selected_for_basin`
  - `n_basins`
  - `n_nodes`
- explain what suspicious values look like

Examples of interpretation:

- too few poses
- too many poses but too few selected
- many selected but no retained basins

### Part 6. Escalating to a difficult heterogeneous-support case

Purpose:

- teach when users should leave the simple high-level path and switch to explicit workflow configuration

Content:

- explain that cluster-on-slab or heterogeneous support systems often need:
  - support pre-relax
  - larger coverage budgets
  - case-scoped exceptions
- demonstrate:
  - `make_adsorption_workflow_config(...)`
  - `make_default_surface_preprocessor(...)`
  - `run_adsorption_workflow(...)`

Recommended case-scoped settings to teach:

- `exhaustive_pose_sampling=True`
- `schedule.pre_relax_selection.max_candidates = 64`
- `cfg.max_selected_primitives = None`
- `adaptive_height_fallback=True`
- `surface_reconstruction_enabled=False`
- support pre-relax before adsorption workflow

Teaching focus:

- these are not repo-wide defaults
- these are escalations for difficult cases only

### Part 7. Real repo case studies

Purpose:

- connect tutorial code to audited repo results

Content:

- point users to:
  - `Pt(211)+Ag4+C6H6`
  - `Pt(211)+Ag4+CO`
- summarize current audited counts from those runs
- explain what they demonstrate about difficult-case coverage

### Part 8. Practical tuning guide

Purpose:

- give users a safe first-response checklist before they start editing internals

Content:

- when to increase pre-relax budget
- when to use exhaustive pose sampling
- when to inspect `raw_site_dictionary.json`
- when to pre-relax support
- when to keep changes case-scoped rather than changing defaults

## Cell Design Principles

The notebook should follow these conventions:

- each major section begins with a markdown cell explaining the purpose
- code cells should be short enough to read without scrolling too much
- after important code cells, include a markdown cell titled conceptually as:
  - "What happened here"
  - "What to inspect next"
  - "If this fails"
- avoid long opaque helper functions unless they improve readability
- avoid burying the key public API call in too much setup code

## Narrative Style

The tone should be:

- direct
- explanatory
- practical
- non-promotional

The notebook should not read like a benchmark report or internal lab memo. It should read like a guided lab manual for a technically serious user.

## Technical Requirements

The notebook should:

- import only current public APIs actually exported by the repo
- align with the README's newly documented recommended usage
- avoid hard-failing when CUDA or MACE is unavailable
- clearly separate:
  - conceptual smoke path
  - production path
  - advanced hard-case path

Where production-only cells require CUDA or MACE, they should be guarded with explicit checks and explanatory markdown.

## Verification Plan

Before claiming completion of the notebook implementation:

1. confirm the notebook file exists at the target path
2. inspect the notebook JSON structure for valid cells
3. run a lightweight parse check with Python on the notebook JSON
4. if practical, execute at least the import/environment-check cells or equivalent extracted code

## Non-Goals

The notebook should not:

- become a full benchmark matrix runner
- duplicate every script in `tools/`
- hide important caveats about downgrade mode
- present difficult-case overrides as safe repo-wide defaults
- replace the existing `examples/full_usage.ipynb`

## Implementation Scope

Implementation after approval should include:

- creating `examples/progressive_teaching_workflow.ipynb`
- ensuring the notebook uses the current recommended public APIs
- optionally adding a short mention in README or `examples/FULL_USAGE.md` if needed, but only if that improves discoverability without expanding scope unnecessarily
