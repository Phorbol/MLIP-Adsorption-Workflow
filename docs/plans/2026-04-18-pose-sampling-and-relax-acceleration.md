# Pt211+Ag4+C6H6 Performance Acceleration Plan

## Context

Fresh profiling for the case-scoped production configuration is recorded in:

- [runs/20260417-234454-pt211-ag4-c6h6/profiling/plan2_stage_profile.json](/mnt/d/Download/trae-research-code/Adsorption-ensemble-Pipline/runs/20260417-234454-pt211-ag4-c6h6/profiling/plan2_stage_profile.json)

Observed wall times:

- `total_wall_seconds = 98.96`
- `slab_pre_relax = 20.21 s`
- `pose_sampling = 17.81 s`
- `pre_relax_selection = 4.14 s`
- `basin_relax = 55.41 s`

The dominant costs are therefore:

1. batch MACE relaxation
2. geometric height search during pose generation

## Root-Cause Summary

### `slab_pre_relax`

This stage is not slow because it fell back to CPU. It ran with:

- `mace_calc_file_batch_relax|cuda|float32|cueq=1`

The main reasons it is still expensive are:

- only one small frame is being relaxed, so GPU utilization is poor
- the backend uses the same batch-relax machinery as multi-frame jobs
- the relax path writes trajectory/log artifacts even for the single-frame slab case
- the optimizer can take up to `200` steps

Interpretation:

- this is mostly fixed overhead plus optimizer iterations
- the current implementation is throughput-oriented, not low-latency-oriented

### `pose_sampling`

The bottleneck is the repeated height-feasibility search in
[adsorption_ensemble/pose/sampler.py](/mnt/d/Download/trae-research-code/Adsorption-ensemble-Pipline/adsorption_ensemble/pose/sampler.py).

Profile highlights:

- `n_solve_height = 10080`
- `n_height_checks = 551800`
- `t_solve_height_s = 14.80 s`
- `t_height_checks_s = 14.55 s`
- `neighborlist_used = false`

Interpretation:

- the cost is overwhelmingly in repeated scaled-distance evaluations
- this case is too small to benefit from the current neighbor-list threshold
- Python control flow is still on the critical path

## Recommended Optimization Order

The safest order is:

1. accelerate the pose-sampling distance kernel
2. add optional low-overhead slab pre-relax mode
3. add primitive-level parallel sampling

This order is preferred because:

- `pose_sampling` is fully deterministic geometry code and easier to validate
- a single-frame slab pre-relax fast path is low-risk and localized
- multiprocessing should be last because it complicates reproducibility and debugging

## Phase 1: Accelerate Height Checks

### Goal

Reduce the cost of `_min_scaled_distance_site_and_surface()` and the loops that call it.

### Proposal

Introduce a compiled fast path for repeated scaled-distance evaluation:

- keep the Python orchestration unchanged
- compile the inner distance kernel with `numba`
- support both:
  - global minimum surface distance
  - site-local minimum over the primitive atoms

### Why This First

- it targets the measured hotspot directly
- it preserves existing search logic and acceptance criteria
- it does not change the scientific semantics of the sampler

### Expected Gain

For this case:

- realistic target: `1.5x` to `2.5x` speedup for `pose_sampling`
- rough wall-time reduction: `6-10 s`

### Validation

Must preserve:

- exact pose count
- exact selected-pose count after pruning
- exact `height_shift_index` / fallback behavior
- exact artifact-level outputs for the fixed seed case

## Phase 2: Add Low-Overhead Single-Frame Slab Relax

### Goal

Reduce fixed overhead in `slab_pre_relax` without changing the MACE model or optimizer target.

### Proposal

Add an optional single-frame relax path for the slab-only pre-relax stage:

- still use CUDA + cuEq
- but disable or minimize:
  - trajectory writing
  - append-stream writing
  - unnecessary batch bookkeeping

This can be exposed as a backend option or a separate lightweight helper used only by case drivers.

### Why This Is Safe

- the slab pre-relax is an upstream preparation step
- this does not need the full multi-frame batch artifact machinery
- behavior can remain numerically identical while reducing I/O and orchestration cost

### Expected Gain

For this case:

- realistic target: `20-40%` reduction for `slab_pre_relax`
- rough wall-time reduction: `4-8 s`

### Validation

Must preserve:

- same backend family: CUDA MACE
- same final relaxed slab geometry within a tight RMSD/energy tolerance
- same downstream site dictionary and basin statistics

## Phase 3: Primitive-Level Parallel Sampling

### Goal

Parallelize pose generation across independent primitives after the compiled kernel work is exhausted.

### Proposal

Split the primitive list into chunks and sample them independently:

- each worker receives:
  - slab geometry
  - adsorbate
  - a primitive subset
  - deterministic RNG offsets
- the main process performs:
  - global pose concatenation
  - final prune
  - artifact ordering

### Why Not First

- more implementation complexity
- higher reproducibility risk
- larger memory and serialization overhead
- gains may be smaller after the kernel hotspot is fixed

### Expected Gain

Case-dependent:

- likely useful on many-core CPU hosts
- limited value on small primitive counts or after aggressive kernel acceleration

## Things Not Worth Doing First

### 1. Python threads

Not preferred as a first optimization because:

- the code is still Python-heavy
- object manipulation and control flow limit thread scaling
- a compiled kernel gives cleaner and more predictable wins

### 2. Lowering global defaults immediately

Not preferred because:

- the user explicitly requested case-scoped behavior previously
- global changes should follow after benchmark evidence

### 3. Replacing the height search logic entirely

Not preferred because:

- the current solver is scientifically interpretable
- changing the search algorithm changes admissible initial-pose semantics
- this should only happen after the current algorithm is fast enough to study

## Benchmark Contract

Every acceleration attempt should be measured on the fixed case:

- support: `Pt(211)+Ag4`
- adsorbate: `C6H6`
- same random seed
- same case driver overrides

Required metrics:

- `slab_pre_relax` wall time
- `pose_sampling` wall time
- `pre_relax_selection` wall time
- `basin_relax` wall time
- `n_pose_frames`
- `n_pose_frames_selected_for_basin`
- `n_basins`
- `n_basins_with_ag_binding`
- exact or tolerance-bounded agreement for artifact-level outputs

## Recommended Immediate Next Step

Implement Phase 1 only:

- add a numba-backed fast path for the distance kernel in the pose sampler
- keep all scientific decision logic unchanged
- rerun the single fixed case
- compare both timing and artifact equivalence

This is the highest signal-to-risk ratio improvement available from the current evidence.
