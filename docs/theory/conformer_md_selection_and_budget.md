# Conformer MD Selection, Budget, and Adsorption Coupling

## Scope

This note formalizes the current `conformer_md` workflow and records the
recommended redesign direction for production use.

The goal is not to describe an idealized future implementation. The document
separates:

- current behavior that is directly verifiable from the repository source code
- externally informed design guidance, using DockOnSurf, CREST/CREGEN, and
  Molclus-like practice as reference points
- concrete parameter/interface changes that should be exposed to users

This document focuses on the free-adsorbate conformer stage that may precede
adsorption ensemble generation.

## Why This Matters

For flexible adsorbates, the quality of adsorption ensembles depends strongly on
which gas-phase conformers are passed downstream into site and pose sampling.

Two requirements are in tension:

- standalone conformer search should be selective and low-noise
- adsorption-coupled conformer search should remain broad enough that
  gas-phase-disfavored conformers are not discarded too early, because surface
  binding can compensate intramolecular strain

Therefore, conformer search needs:

- an explicit final conformer budget
- a clearly defined structure-diversity metric
- different default filtering profiles for isolated search and
  adsorption-coupled search

## Current Code Path

The current entry points are:

- `adsorption_ensemble.conformer_md.config`
- `adsorption_ensemble.conformer_md.pipeline`
- `adsorption_ensemble.conformer_md.xtb`
- `adsorption_ensemble.conformer_md.selectors`
- `adsorption_ensemble.workflows.adsorption`
- `adsorption_ensemble.workflows.flex_sampling`

At a high level, the implemented workflow is:

1. Run one or more xTB GFN-FF MD trajectories for the isolated adsorbate.
2. Concatenate all MD frames.
3. Compute raw descriptors on all frames.
4. Preselect a subset with `fps`, `fps_pca_kmeans`, or `kmeans`.
5. Apply a loose stage.
6. Apply a final stage.
7. Return the selected conformers to the adsorption workflow.

However, several details of the current implementation are important.

### Current xTB MD stage

Current defaults in `XTBMDConfig` are:

- `temperature_k = 400`
- `time_ps = 100`
- `step_fs = 1`
- `dump_fs = 50`
- `seed = 42`
- `gfnff = True`
- `n_runs = 1`

The runner writes `input.xyz`, generates an `md.inp`, executes:

`xtb input.xyz --gfnff --md --input md.inp --omd`

and reads `xtb.trj` back as the trajectory.

Current limitations:

- there is no explicit propagation of charge and spin into the xTB command
- `n_runs` repeats the same config unless the caller manually changes the seed
- partial success is accepted if `xtb.trj` exists and is readable

### Current descriptor stage

Two raw descriptor backends exist:

- `geometry`
- `mace`

Raw-frame preselection can therefore already use MACE descriptors. However,
after preselection the pipeline switches to geometric pair-distance vectors for
the loose and final filtering stages.

This means the descriptor used for:

- raw diversity selection

is not necessarily the same descriptor used for:

- loose-stage deduplication
- final-stage deduplication

### Current selection semantics

`SelectionConfig` currently exposes:

- `preselect_k`
- `mode`
- `energy_window_ev`
- `rmsd_threshold`
- loose/final filter strategy overrides
- FPS/PCA/KMeans control parameters

There is no explicit parameter representing:

- the required final conformer budget

In practice:

- `preselect_k` is a pre-relax cap
- final cardinality is whatever survives energy-plus-diversity filtering

That is acceptable for offline exploratory use, but it is not production-grade
for a coupled adsorption workflow because downstream compute cost scales with the
number of retained conformers.

There is also a current implementation detail that materially affects the
semantics of preselection:

- the raw-frame `_preselect` call in the pipeline currently receives a zero
  energy array

Therefore:

- `fps_pca_kmeans` cannot currently interpret "pick the lowest-energy member of
  each reduced cluster" using actual frame energies
- `kmeans` preselection likewise does not currently select using actual MD-frame
  energies

In other words, the current preselection stage is diversity-driven but not
meaningfully energy-aware, even though the interface suggests otherwise.

### Current "RMSD" behavior

The current `RMSDSelector` is not a Kabsch-aligned Cartesian RMSD filter.

It is a Euclidean threshold in descriptor space:

- when used in the adsorption multistage schedule, it operates on the chosen
  stage descriptor
- in `conformer_md`, after preselection it operates on geometric pair-distance
  vectors

Therefore the current name `rmsd_threshold` is semantically misleading in the
`conformer_md` pipeline. It is acting as a structure-diversity threshold, not a
literal Cartesian RMSD.

### Current relax-stage behavior

The pipeline supports three backends:

- `identity`
- `mace_energy`
- `mace_relax`

Only `mace_relax` actually updates coordinates.

When `mace_energy` is used, the stage only assigns energies and returns the
input geometries unchanged. For production conformer search this creates a
serious ambiguity:

- the pipeline labels the stages as loose/refine relax
- but if `mace_energy` is selected, no geometry relaxation occurs

### Current energy semantics

The conformer pipeline stores and compares `energies_ev`, but the current MACE
backends in this module return energy per atom for each frame.

For conformer ranking, the physically meaningful quantity is the total energy of
the molecule, not the per-atom energy. Using per-atom energy for same-composition
comparisons is numerically monotonic in principle, but it obscures the intended
semantics and can easily leak into user interpretation or mixed-size workflows.

## Current Publicly Exposed Parameters

The current public knobs are spread across:

- `ConformerMDSamplerConfig`
- the `conformer-md` CLI
- adsorption workflow schedules and flexible-budget heuristics

### `conformer-md` CLI

The current CLI exposes:

- MD controls:
  - `--temperature-k`
  - `--time-ps`
  - `--step-fs`
  - `--dump-fs`
  - `--seed`
  - `--n-runs`
- preselection controls:
  - `--preselect-k`
  - `--mode`
  - `--pca-variance-threshold`
  - `--fps-pool-factor`
- final filtering controls:
  - `--energy-window-ev`
  - `--rmsd-threshold`
- backend controls:
  - `--descriptor-backend`
  - `--relax-backend`
  - `--mace-model`
  - `--mace-device`
  - `--mace-dtype`
  - `--mace-max-edges`
  - `--mace-workers`
  - `--mace-layers-to-keep`
  - `--mace-energy-key`
  - `--mace-head-name`
- relax-stage controls:
  - `--loose-maxf`
  - `--loose-steps`
  - `--refine-maxf`
  - `--refine-steps`

### What is still missing from the public API

For production use, the following concepts should be explicit and user-visible:

- `target_final_k`
- `selection_profile`
- `metric_backend`
- `metric_threshold`
- `pair_energy_gap_ev`
- `use_total_energy`
- `seed_mode`
- `charge`
- `spin_multiplicity`
- `mace_metric_dtype`
- `mace_metric_enable_cueq`

Without these, the user cannot cleanly express:

- "I want 8 final representatives, no more"
- "I want broad energy retention for adsorption seeding"
- "Use MACE features for structural diversity, not geometric distances"

## Comparison With Reference Workflows

## DockOnSurf

DockOnSurf is highly relevant because it explicitly decomposes adsorption
screening into chemically meaningful subspaces: conformers, anchoring points,
sites, orientations, and optional proton dissociation.

Relevant input semantics documented by DockOnSurf:

- `num_conformers` controls the number of raw conformers to generate before
  optimization; it is explicitly not the final number of structures to optimize.
- `select_magns` chooses the quantity used to select conformers from the
  isolated stage, with documented options `energy` and `MOI`.
- `confs_per_magn` explicitly controls how many conformers are retained per
  selection magnitude.
- `sample_points_per_angle` makes the orientational budget explicit, with the
  documented `n^3 - n(n-1)` combinatorics after redundancy filtering.
- `collision_threshold` and `min_coll_height` make clash screening explicit.
- `max_structures` provides an explicit cap on generated adsorption structures.

The main design lesson is not that this repository should copy DockOnSurf's
exact interface. The important lesson is that DockOnSurf distinguishes:

- raw generation budget
- selection criterion
- selected representative count
- adsorption-structure generation budget

These are first-class parameters, not hidden consequences of other thresholds.

## CREST and CREGEN

CREST is relevant because CREGEN makes the ensemble sorting thresholds explicit.

Documented CREGEN defaults include:

- `--ewin`: energy window, with 6 kcal/mol default for conformational searches
- `--rthr`: RMSD threshold, default 0.125 Angstrom
- `--ethr`: energy threshold between conformer pairs, default 0.05 kcal/mol
- `--bthr`: lower bound for rotational-constant threshold, default 1%

The CREGEN perspective is useful for this repository even if we do not adopt
Cartesian RMSD:

- the energy window and structural threshold are separate controls
- the pair-energy threshold and the global window are also separate controls
- topology mismatches are treated explicitly

The main transferable design principle is:

- use multiple filters with distinct semantics, not one overloaded
  `energy_window_ev + rmsd_threshold` pair for every use case

## Molclus-like practice

Molclus is relevant mainly as a representative of practical conformer
generation-and-clustering workflows used by computational chemists.

In the literature, Molclus is commonly used with:

- broad raw generation
- semiempirical or low-level pre-optimization
- energy-threshold plus geometry-threshold clustering
- an explicit top-k handoff to higher-level optimization

Representative literature examples report workflows such as:

- clustering on `0.5 kcal/mol` energy and `0.25 Angstrom` geometry thresholds
- keeping the top `200` clusters for a first refinement
- then keeping the top `10` for higher-level optimization

The transferable lesson here is again budget semantics:

- raw candidates
- clustered representatives
- promoted top-k for expensive refinement

should be explicitly separated.

## What The Current Repository Is Doing Wrong

The issues below are specific and source-verifiable.

### 1. There is no explicit final conformer budget

`preselect_k` is explicit, but final `k` is emergent.

For production adsorption workflows this is inadequate because:

- site and pose sampling cost grows roughly linearly with the number of
  conformers
- two cases with the same advertised settings can produce very different
  downstream workloads

### 2. The current `rmsd_threshold` name is misleading

The current user-visible name suggests Kabsch-aligned Cartesian RMSD, but the
implemented behavior is descriptor-space Euclidean filtering.

For `conformer_md`, this should be renamed to a metric-agnostic term, for
example:

- `structure_metric_threshold`
- `diversity_threshold`

### 3. The current post-preselection filter should not be geometric by default

For this repository, using MACE invariant features is more internally coherent
than switching back to geometric pair-distance vectors.

Reasons:

- the adsorption workflow already relies on MACE-based structure comparison
- pair-distance vectors are sensitive to atom-count scaling and descriptor-size
  effects
- a chemistry-aware learned invariant is a better similarity proxy than a pure
  distance list for the intended downstream use

If MACE is available, the default conformer diversity metric should therefore be
MACE-based.

### 4. The current preselection API is semantically inconsistent

The selection interface exposes energy-aware modes, but raw-frame preselection
currently does not use actual per-frame energies.

This should be corrected in one of two ways:

- either pass real frame energies into preselection
- or explicitly rename the current behavior as diversity-only preselection

Keeping the current names while providing zero energies is misleading.

### 5. Energy-window defaults should depend on use case

A standalone conformer search and an adsorption-coupled conformer search do not
have the same objective.

Standalone search prefers:

- sharp filtering
- population-relevant low-energy representatives
- low redundancy

Adsorption-coupled search prefers:

- broader retention
- allowing moderately strained gas-phase conformers to survive
- avoiding premature elimination of surface-competent motifs

Therefore a single default `energy_window_ev` is not physically well-posed.

### 6. The current pipeline mixes three different meanings of "selection"

The following should be distinguished:

- diversity-oriented preselection from raw MD frames
- post-relax duplicate removal
- final representative budgeting for downstream adsorption

These are not the same operation and should not share the same default
thresholds.

### 7. `n_runs` should imply seed diversification

If `n_runs > 1`, the implementation should not silently repeat the same seed.
The interface needs a documented seed policy.

### 8. Charge and spin must be explicit

For realistic adsorbates, especially ions, radicals, and proton-transfer-active
systems, conformer generation without explicit charge and spin propagation is not
production-safe.

## Recommended Redesign

## Design principle

The conformer stage should expose three layers of controls:

1. raw exploration budget
2. structural deduplication semantics
3. final handoff budget to adsorption

These should be explicit in both Python and CLI APIs.

## Recommended parameter model

The following interface is recommended.

### Budget parameters

- `raw_md_budget`
  - meaning: total amount of low-level exploration
  - components:
    - `temperature_k`
    - `time_ps`
    - `step_fs`
    - `dump_fs`
    - `n_runs`
    - `seed_mode`

- `preselect_k`
  - meaning: maximum number of frames promoted from raw MD into the expensive
    post-MD scoring/filtering stages

- `target_final_k`
  - meaning: explicit upper bound on the number of conformers returned by
    `ConformerMDSampler`

- `promotion_k`
  - optional synonym if a multi-stage refine ladder is introduced later

### Metric parameters

- `metric_backend`
  - allowed values:
    - `mace_global`
    - `geometry_pairdist`
    - `mace_global_fp64`
  - recommended default:
    - `mace_global_fp64` when MACE is available

- `metric_threshold`
  - descriptor-space threshold for structural diversity

- `metric_threshold_mode`
  - allowed values:
    - `fixed`
    - `auto_from_distribution`

- `mace_metric_dtype`
  - recommended default: `float64`

- `mace_metric_enable_cueq`
  - recommended default: `False`

### Energy parameters

- `energy_window_ev`
  - global energy window relative to the lowest conformer retained in the stage

- `pair_energy_gap_ev`
  - minimum energy separation required before two geometrically similar
    structures are both kept

- `use_total_energy`
  - recommended default: `True`

### Workflow profile parameter

- `selection_profile`
  - allowed values:
    - `isolated_strict`
    - `adsorption_seed_broad`
    - `manual`

This is the cleanest way to give users robust defaults without forcing them to
manually tune multiple thresholds.

## Recommended profile defaults

The values below are recommendations for this repository design. They are not
claims of universal physical optimality.

### Profile A: `isolated_strict`

Intended use:

- user wants a compact, low-noise gas-phase conformer ensemble
- user may later run higher-level gas-phase refinement
- adsorption is not the immediate downstream bottleneck

Recommended defaults:

- `preselect_k = max(64, 4 * target_final_k)`
- `target_final_k = 12`
- `metric_backend = mace_global_fp64`
- `energy_window_ev = 0.20`
- `pair_energy_gap_ev = 0.02`
- `metric_threshold_mode = auto_from_distribution`
- `relax_backend = mace_relax`

Interpretation:

- keep the ensemble compact
- aggressively remove near-duplicates
- bias toward low-energy representatives

### Profile B: `adsorption_seed_broad`

Intended use:

- conformers are only an upstream seed set for adsorption structure generation
- surface binding may favor geometries that are not gas-phase minima

Recommended defaults:

- `preselect_k = max(96, 6 * target_final_k)`
- `target_final_k = 8`
- `metric_backend = mace_global_fp64`
- `energy_window_ev = 0.60`
- `pair_energy_gap_ev = 0.01`
- `metric_threshold_mode = auto_from_distribution`
- `relax_backend = mace_relax`

Interpretation:

- allow a materially wider energy range than in isolated-only searches
- keep the final handoff budget explicit and modest
- preserve broader motif diversity before adsorption multiplies the workload

## Why `target_final_k = 8` is the recommended adsorption-coupled default

This repository performs:

- conformer generation
- site sampling
- pose sampling
- relaxation
- basin deduplication

The conformer count is therefore multiplied by downstream surface and pose
budgets. A default `target_final_k = 8` is a pragmatic compromise:

- larger than a trivial single-minimum handoff
- small enough that downstream work does not explode for typical stepped and
  heterogeneous surfaces

For difficult polyfunctional adsorbates a heuristic auto-budget is still useful.
The recommended auto policy is:

- rigid or near-rigid adsorbates: `1`
- mildly flexible adsorbates: `4`
- clearly flexible polyfunctional adsorbates: `8`
- very large or highly branched adsorbates: `12`

Even when auto mode is used, the resolved value of `target_final_k` must be
written to metadata so the run budget is explicit and reproducible.

## Why MACE feature distance is preferred here

The user requirement for this repository is not:

- geometrically exact Cartesian RMSD deduplication

It is:

- chemically meaningful deduplication for downstream adsorption sampling

For this purpose, a MACE invariant descriptor is preferable to Kabsch-aligned
RMSD or pair-distance vectors because:

- it is already aligned with the MLIP family used elsewhere in the project
- it is translation and rotation invariant by construction
- it better reflects local chemical environment than a pure geometry-only vector
- it integrates naturally with the basin-dedup direction already adopted in this
  repository

Recommended rule:

- if MACE is available, use MACE features for conformer diversity filtering
- only fall back to geometric pair distances when MACE is unavailable

## FP64 and cuEq recommendations for structure comparison

For feature extraction used only for structure comparison:

- use `float64`
- set `enable_cueq = False`

Reasoning:

- reproducibility and threshold stability matter more than maximum throughput
- conformer similarity thresholds are sensitive to small descriptor differences
- this stage is generally not the dominant wall-time cost compared with
  downstream adsorption relaxation

This does not imply the same defaults for heavy batch relaxation, where
`float32 + cuEq` can remain a throughput-oriented choice.

## Recommended algorithmic changes

## 1. Make the final budget explicit

Add `target_final_k` to `SelectionConfig` and CLI.

Required behavior:

- after energy-window and metric-threshold filtering, if the surviving set is
  larger than `target_final_k`, apply a final rank-and-thin step
- the final rank-and-thin step should preserve both low energy and descriptor
  diversity

Recommended implementation:

- sort candidates by total energy
- take the lowest-energy conformer as the first representative
- iteratively add the conformer that maximizes the minimum descriptor distance to
  the selected set, subject to remaining inside the active energy window

This makes the final budget deterministic and diversity-aware.

## 2. Replace the current geometric post-preselection metric default

Current behavior in `conformer_md` should be changed from:

- raw preselection on MACE or geometry
- then loose/final filtering on geometric pair-distance vectors

to:

- use the same metric family consistently across preselect, loose filter, and
  final filter unless the user explicitly requests otherwise

Recommended default:

- `metric_backend = mace_global_fp64`

## 3. Rename `rmsd_threshold`

Recommended rename:

- from `rmsd_threshold`
- to `structure_metric_threshold`

Backward compatibility can be preserved by:

- keeping `rmsd_threshold` as a deprecated alias
- documenting that it is not a Kabsch RMSD in this repository

Current repository status:

- the config now supports `structure_metric_threshold` as the preferred name
- `rmsd_threshold` remains as a compatibility alias

## 4. Separate isolated and adsorption-coupled defaults

Introduce a new profile resolver:

- `resolve_conformer_selection_profile(profile, molecule, workflow_context)`

This function should set:

- `energy_window_ev`
- `pair_energy_gap_ev`
- `target_final_k`
- `preselect_k`
- `metric_backend`
- `metric_threshold_mode`

based on whether the conformer output is:

- final output
- or an upstream seed set for adsorption

## 5. Use total energy semantics

Conformer ranking, filtering, and metadata should all use total energy for the
isolated molecule.

If a backend naturally reports per-atom energy, convert it to total energy
before conformer selection metadata is written.

## 6. Make seed policy explicit

Recommended values:

- `seed_mode = fixed`
  - preserve exact reproducibility
- `seed_mode = increment_per_run`
  - use `seed + run_index`
- `seed_mode = hashed`
  - use a stable hash of `(seed, job_name, run_index)`

Recommended default:

- `increment_per_run`

## 7. Pass charge and spin through the conformer pipeline

The conformer API should accept:

- `charge`
- `spin_multiplicity`

and propagate them into:

- xTB calls
- MACE config construction when needed
- metadata and output files

## What Should Change In The Adsorption Workflow

The adsorption workflow should not only toggle:

- `run_conformer_search`

It should also record the intended conformer usage mode.

Recommended new field:

- `conformer_usage_mode`
  - allowed values:
    - `disabled`
    - `isolated_analysis`
    - `adsorption_seed`

Then the schedule builder or workflow preset can resolve the correct default
selection profile.

This is cleaner than overloading one boolean with several different semantics.

## Minimum Production-Safe Changes

The changes below are the minimum set that should be treated as production
blocking for this repository.

1. Add explicit `target_final_k`.
2. Use MACE-based structure filtering by default when MACE is available.
3. Split defaults into `isolated_strict` and `adsorption_seed_broad`.
4. Rename or deprecate the misleading `rmsd_threshold`.
5. Use total-energy semantics in conformer selection.
6. Diversify seeds across `n_runs`.
7. Propagate charge and spin.
8. Default MACE comparison to `float64 + cueq off`.

## Recommended Metadata Additions

Every conformer run should record:

- `selection_profile`
- `resolved_preselect_k`
- `resolved_target_final_k`
- `metric_backend`
- `metric_threshold`
- `metric_threshold_mode`
- `energy_window_ev`
- `pair_energy_gap_ev`
- `energy_semantics`
- `seed_mode`
- `per_run_seeds`
- `charge`
- `spin_multiplicity`

This is necessary for reproducibility and for profiling whether the conformer
stage is overly restrictive or overly permissive.

## Practical Summary

The repository should move toward the following behavior:

- use MACE feature distance for conformer diversity by default
- stop calling the threshold `rmsd_threshold` when it is not RMSD
- expose an explicit final conformer budget
- use stricter defaults for standalone gas-phase conformer search
- use broader defaults for adsorption-coupled conformer seeding
- default structure-comparison MACE inference to `float64 + cueq off`

This keeps the workflow aligned with the project's broader design direction:

- chemically aware learned descriptors for structure comparison
- explicit run budgets
- robust defaults that minimize user tuning

## External References

Primary or near-primary references consulted for this note:

- DockOnSurf input manual:
  - https://dockonsurf.readthedocs.io/en/latest/inp_ref_manual.html
- DockOnSurf project and paper landing page:
  - https://chemrxiv.org/engage/chemrxiv/article-details/60c73d0e4c8919d2ebad1b54
- CREST keyword documentation:
  - https://crest-lab.github.io/crest-docs/page/documentation/keywords.html
- CREST ensemble sorting guide:
  - https://crest-lab.github.io/crest-docs/page/examples/example_2.html

Additional literature examples used only as Molclus-like workflow evidence:

- Molclus usage example with `0.5 kcal/mol` and `0.25 Angstrom` thresholds and
  explicit `top 200 -> top 10` promotion:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9482987/
- Molclus usage example emphasizing explicit geometry and energy thresholds:
  - https://pubs.rsc.org/en/content/articlehtml/2025/cp/d4cp04184d
