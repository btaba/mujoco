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

## Collider selection flags

Two option flags select the box-box collider at runtime (checkboxes in the
simulate viewer's Option panel, or XML):

- default: the current specialized collider
- `<flag boxboxlegacy="enable"/>`: the pre-rewrite legacy collider
- `<flag boxbox="disable"/>`: route box-box pairs to the general convex
  pipeline (GJK/EPA); takes precedence over `boxboxlegacy`

## Stacking demo: `stacking_demo/`

`tower30.xml` (30-cube aligned tower) and `perturbed.xml` (three jittered
towers). Load in the viewer, run, and flip the flags. Measured with
`stack_bench.py` (settle velocity, lateral drift, fallen boxes, contacts,
solver iterations, microseconds per step -- the ccd-manifold-depth
proposal's stack metrics), 4 simulated seconds:

| scenario  | mode   | settle_vel | drift  | fallen | us/step |
|-----------|--------|-----------:|-------:|-------:|--------:|
| cubes30   | new    |    3.4e-09 | 0.0000 |      0 |    83.4 |
| cubes30   | legacy |    1.3e+01 | 1.4208 |     27 |    85.0 |
| cubes30   | gjk    |    2.7e-03 | 0.0001 |      0 |   108.8 |
| perturbed | new    |    1.4e-02 | 0.0090 |      0 |    93.7 |
| perturbed | legacy |    6.8e-03 | 0.0247 |      0 |   120.1 |
| perturbed | gjk    |    4.0e+00 | 0.1488 |      0 |   168.0 |
| plates    | new    |    2.7e-02 | 0.0002 |      0 |    75.4 |
| plates    | legacy |    3.4e-04 | 0.0016 |      0 |    66.5 |
| plates    | gjk    |    2.2e-01 | 0.0071 |      1 |   146.9 |

The legacy collider collapses the tall aligned tower (27 of 30 cubes
fallen); the convex pipeline keeps towers up but never comes to rest on the
perturbed scenario and drops a plate, at 1.5-2x the step cost.

## Per-point depth: `plate_pair.py`

Mirrors the ccd-manifold-depth proposal's plate-pair analysis: a
near-parallel tilted plate pair where the true gap varies across the
contact patch. The current convex pipeline stamps the single EPA depth on
every clipped vertex: on the deep pose the far corner's depth is off by
1.4e-4 (7%), and on the grazing pose it reports penetration at corners
that are separated -- the fabricated-penetration artifact that proposal
corrects. The box-box collider (and the legacy one) report per-point
depths matching the analytic gap to float precision in both poses, so the
proposal's fix converges the convex pipeline toward what the specialized
collider already produces, at higher cost.

## Head-to-head vs ccd-manifold-depth (Yuval's per-point depth patch)

Commit 25745832 from yuvaltassa/ccd-manifold-depth is cherry-picked onto
this branch, so the `boxbox="disable"` mode IS the patched convex pipeline.
All numbers below are measured against it (not the unpatched upstream).

The patch works as proposed: uniform depth stamping and fabricated
penetration are gone from the EPA manifold (plate_pair.py now shows
per-point-correct gjk depths), aligned towers settle to machine stillness,
and the plates scenario improves 500x in settle velocity.

stack_bench.py, 4 simulated seconds, after the patch:

| scenario  | mode | settle_vel | drift  | fallen | us/step |
|-----------|------|-----------:|-------:|-------:|--------:|
| cubes30   | new  |    3.4e-09 | 0.0000 |      0 |    89.8 |
| cubes30   | gjk  |    3.5e-09 | 0.0000 |      0 |   101.8 |
| perturbed | new  |    1.4e-02 | 0.0090 |      0 |   100.2 |
| perturbed | gjk  |    1.1e-01 | 0.0131 |      0 |   154.4 |
| plates    | new  |    2.7e-02 | 0.0002 |      0 |    80.9 |
| plates    | gjk  |    4.2e-04 | 0.0019 |      0 |    83.0 |

Depth-vs-exact percentiles (search_gjk_failures.py, 10k poses/case):

| case      | collider | median  | p99     | max     |
|-----------|----------|--------:|--------:|--------:|
| cube_cube | box-box  | 1.4e-17 | 3.4e-04 | 9.6e-04 |
| cube_cube | gjk      | 1.9e-09 | 2.1e-09 | 8.1e-07 |
| aniso     | box-box  | 6.9e-18 | 5.2e-05 | 2.8e-04 |
| aniso     | gjk      | 8.5e-10 | 2.0e-09 | 2.1e-08 |

Where each wins, after the patch:
- box-box: median depth exactness (8 orders), step cost (+13-17% towers,
  +54% perturbed), perturbed-stack quietness (7.6x lower settle velocity,
  fewer contacts and solver iterations).
- patched gjk: depth-error p99/max (its 1e-9 tolerance beats box-box's
  deliberate 5% face-preference tail, which trades near-tie depth for the
  stack stability above), and plates settle velocity (60x quieter; the
  4-contact cap is the suspected cause, under investigation).
- The patch remains valuable for general convex pairs (mesh-mesh), where
  no specialized collider exists.
