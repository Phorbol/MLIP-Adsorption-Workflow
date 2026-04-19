# Initial Pose Height Search

## Scope

This note formalizes the initial-height search used by
`adsorption_ensemble.pose.sampler.PoseSampler` for adsorption-pose generation.
It describes the current implementation rather than an idealized future model.

The target problem is:

- input:
  - a slab `S`
  - an adsorbate geometry `A`
  - a site primitive `p = (c, n, t1, t2, I_site)`
  - an orientation sample `R`
- output:
  - a placement height `h` such that the translated adsorbate
    `R(A) + c + h n`
    is geometrically admissible as an initial guess for relaxation

Here:

- `c` is the site center
- `n` is the site normal
- `I_site` is the set of surface atoms defining the primitive

## Geometric Model

For a rotated adsorbate with atom positions `x_i` and surface atom positions `y_j`,
the placed coordinates at height `h` are

`x_i(h) = x_i + c + h n`

For adsorbate atom `i` and surface atom `j`, define the scaled distance

`d_ij(h) = ||x_i(h) - y_j||_MIC / (r_i + r_j)`

where:

- `||.||_MIC` is the minimum-image distance under slab PBC
- `r_i, r_j` are covalent radii

Two reduced distances are used:

- site-local minimum
  - `d_site(h) = min_{i, j in I_site} d_ij(h)`
- global surface minimum
  - `d_surf(h) = min_{i, j in surface} d_ij(h)`

## Feasibility Constraints

The sampler uses two geometric constraints.

### 1. Hard clash constraint

The placement must satisfy

`d_surf(h) >= tau_clash`

This prevents obviously overlapping initial guesses.

### 2. Site-contact constraint

The placement must also satisfy

`d_site(h) >= tau_target`

where `tau_target = max(tau_height, tau_clash)`.

For low-coordination sites, the algorithm additionally requires the contact not
to drift too far away from the intended site. A contact metric is defined as:

- for `1c/2c` sites:
  - `m_contact(h) = d_site(h)`
- for `3c+` sites:
  - `m_contact(h) = d_surf(h)`

The accepted height must satisfy

`m_contact(h) <= tau_target + delta_contact`

where `delta_contact` is the configured site-contact tolerance plus the
adsorbate-shape-specific allowance for linear molecules.

Interpretation:

- the lower bound avoids penetration
- the upper bound avoids placing the adsorbate so high that it is no longer
  meaningfully associated with the chosen site

## Height Search Problem

For each orientation sample, the code solves

`find the smallest h in [h_min, h_max] such that`

- `d_surf(h) >= tau_clash`
- `d_site(h) >= tau_target`
- and, when enforced, `m_contact(h) <= tau_target + delta_contact`

This is a one-dimensional constrained feasibility problem along the site normal.

## Current Numerical Procedure

The implementation in `PoseSampler._solve_height()` uses a coarse-to-fine search.

### Step 1. Lower bound selection

The search starts from

`h_lo = max(config.min_height, h_pref(molecule_shape, site_coordination))`

where `h_pref` is a conservative floor for linear adsorbates on bridge/hollow
sites.

### Step 2. Tau schedule

The solver loops over a schedule

`tau_height in height_taus`

The first tau that yields a valid solution is accepted.

### Step 3. Upward probing

Starting from `h_lo`, the code increments height by `height_step` until it finds
the first height that satisfies the lower-bound constraints.

If no feasible height exists in `[h_lo, h_max]`, the current tau fails.

### Step 4. Contact-window screening

For `1c/2c` sites, the first feasible height is rejected if it is already too
far from the intended contact window.

### Step 5. Bisection refinement

After a feasible bracket is found, the code bisects between the last infeasible
and first feasible heights for `height_bisect_steps`.

The result is the smallest numerically admissible height under the current tau.

## Adaptive Height Fallback

The case-scoped fallback added on 2026-04-18 is a primitive-local rescue step.

It is only triggered when:

- one primitive produces zero accepted poses after the normal search
- and `adaptive_height_fallback=True`

The fallback does not change global defaults.

### Fallback objective

It solves a relaxed version of the original problem:

`find h >= h_start such that d_surf(h) >= tau_clash`

and, when the contact window is enforced,

`m_contact(h) <= tau_target + delta_contact + delta_fallback`

where `delta_fallback` is an extra slack term.

### Fallback interval

The search starts at

- `max(h_pref, h_base + step_fallback)` if the base solver found a nominal height
- otherwise `h_pref`

and continues up to

`max(h_start, h_max) + h_extra_max`

with step size

`max(height_step, step_fallback)`

### Interpretation

The fallback assumes some primitives fail not because the site is invalid, but
because the first admissible contact along the normal is slightly above the
default search window or slightly outside the nominal contact tolerance.

This is a rescue for difficult geometries, not a replacement for the main
height solver.

## Why The Search Is Expensive

The dominant cost comes from repeated evaluations of

- `d_site(h)`
- `d_surf(h)`

for many combinations of:

- primitive
- orientation
- azimuth
- tangent shift
- trial height
- tau schedule

For the profiled `Pt(211)+Ag4+C6H6` production case:

- `n_solve_height = 10080`
- `n_height_checks = 551800`

So the practical cost driver is not final deduplication, but repeated
one-dimensional geometric feasibility tests.

## Practical Consequences

- The current solver is conservative: it prefers the smallest admissible height.
- The search is physically interpretable: every accepted pose is explicitly
  non-overlapping and still associated with its intended site.
- Upstream pose coverage is sensitive to:
  - `h_max`
  - `height_step`
  - `height_taus`
  - `site_contact_tolerance`
  - whether adaptive fallback is enabled

## Limitations

- The method is purely geometric and does not use local energy gradients.
- The contact window is heuristic, not derived from a force field.
- The search is serial and Python-heavy in the current implementation.
- The fallback may recover valid candidates, but it can also admit very high
  initial poses if the search window is made too permissive.

## Optimization Directions

If performance becomes a priority, the cleanest acceleration targets are:

- JIT or native kernels for repeated scaled-distance evaluation
- primitive-level parallel sampling on CPU
- reducing the number of trial heights via better bracketing
- replacing the linear upward scan with a monotonic root/bracket search where
  the geometry permits it
