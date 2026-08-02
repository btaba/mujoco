# Box-box rewrite: validation and benchmark assets

Companion assets for the box-box collider rewrite. Everything here compares
three quantities per contact pair:

- **box-box**: `mjc_BoxBox` on a box pair,
- **gjk/epa**: `mjc_Convex` on an identical box-mesh pair (8-vertex hull),
- **exact depth**: the maximum support separation over the 15 candidate axes
  (6 face normals, 9 edge cross products), which attains the true penetration
  depth for a box pair; computed independently in numpy at float64.

## Accuracy: `failure_cases/` + `verify_failure_cases.py`

`failure_cases/case*.xml` pose both pairs at configurations (via keyframe
`failure`) where GJK/EPA's reported depth misses the exact depth by orders of
magnitude more than box-box does. Load any case in the viewer to see the
configuration, or run:

    python verify_failure_cases.py

which prints the exact depth and each collider's deepest contact. Regenerate
the models with `make_failure_cases.py`; search for new poses with
`search_gjk_failures.py <case> <n_poses>`, which also prints error
percentiles. Representative sweep over ~30k contacts:

- box-box error vs exact depth: median 1e-17 (machine precision), p90 5e-17.
  The p99 tail (up to ~2e-3) is the documented five-percent face-preference
  band on near-tie deep overlaps, a stack-stability design choice.
- GJK/EPA error: median ~2e-9 (its iterative tolerance), up to ~6e-5 on
  anisotropic boxes, pose-dependent.
- Contact existence agrees on every sampled pose.

## Performance: `perf/` + `perf/benchmark.sh`

Three box-dominated scenes: `pile.xml` (100 boxes falling into a corner),
`towers.xml` (six 10-cube towers at rest), `plates.xml` (nine thin-plate
stacks, the aspect ratio that stressed the previous implementation). Run:

    ./perf/benchmark.sh <old-ref> [nstep]

which builds the reference revision in a temporary worktree, builds the
current checkout, and reports testspeed's microseconds per step for both on
each scene.
