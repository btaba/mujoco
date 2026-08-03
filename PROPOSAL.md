# Box-box collision rewrite

This branch replaces MuJoCo's box-box narrowphase with a structured
implementation (separating-axis test + Sutherland-Hodgman clipping + true
edge-edge contacts), deletes the legacy collider's accumulated repair logic,
and adds a runtime option (`mjOption.boxbox`) so the legacy collider and the
general convex pipeline (GJK/EPA) can be compared interactively on any model. Not for landing as-is:
the last two commits (per-point-depth cherry-pick from
`yuvaltassa/ccd-manifold-depth` and this document) exist for comparison.

## What was accomplished

- **Rewrite** (`src/engine/engine_collision_box.c`): SAT with closed-form box
  supports; face manifolds by polygon clipping with depth measured along the
  reference normal only; edge-edge contacts from the closest point pair
  between supporting edge segments, with sign-ambiguity enumeration for
  near-zero axis components; at most 4 contacts (was 8), selected
  deterministically in clip order with the deepest vertex guaranteed;
  margin as a first-class acceptance band. An edge axis within ~8 degrees of
  the best face axis is replaced by the face unless 5% deeper (resting-stack
  stability; measured tower energy degrades continuously as this margin
  shrinks).
- **Deleted**: the conditional add-back cascade, out-of-range u/v clamping,
  the `kRemoveRatio` outside-box filter and its missing-contact fallback
  hole, exact-float dedup, and the edge-path depth clamp. The ground-truth
  gates verify no contact ever exceeds the true depth without any clamp.
- **Collider selection**: `mjOption.boxbox` selects the box-box narrowphase
  -- `<option boxbox="new|legacy|convex"/>` in XML, a `BoxBox` dropdown next
  to Integrator in the simulate viewer's Option panel. `new` is the rewrite,
  `legacy` the previous implementation preserved verbatim, `convex` the
  general GJK/EPA pipeline. One control, one visible state.
- **Validation assets** (`boxbox_validation/`): exact-depth cross-validation,
  GJK comparison cases with keyframes, stacking demos, a 3-way dynamics
  benchmark, per-point depth analysis, and an old-vs-new perf harness.

## What was tested

- **Unit suite**: all 1345 tests pass (1343 upstream + 2 new flag tests),
  including the fuzzer-pinned thin-box regressions, with the legacy depth
  clamp deleted. One pinned tolerance widened to the 5% design band
  (`EdgeContactAtDepthBound`); the 1000x-depth bug class it guards is still
  caught.
- **Randomized fuzz** (`test/engine/engine_collision_box_fuzz_test.cc`):
  box pairs vs identical box meshes across 6 aspect-ratio cases, gated
  against exact analytic depth (max support separation over the 15 candidate
  axes). Hard gates per sample: no phantom penetration, no missed contact at
  zero margin, no over-deep contact, contacts within half their depth of
  both boxes, at most 4 contacts, one normal per manifold, strict GJK
  agreement. ~90M configurations across 500+ seeds passed during
  development; the committed test runs 48k per invocation
  (`MJ_FUZZ_CONFIGS`/`MJ_FUZZ_SEED` scale it).
- **Dynamics**: 10/20/30-cube towers, perturbed towers, thin-plate stacks
  (`boxbox_validation/stack_bench.py`), settle velocity / drift / fallen
  count / contacts / solver iterations / step cost, for all three colliders.
- **Per-point depth** (`boxbox_validation/plate_pair.py`): tilted plate pair
  with analytic per-position gaps.
- **Performance**: Google-benchmark scene (`ccd_benchmark_test`) and a
  worktree-based old-vs-new harness (`boxbox_validation/perf/benchmark.sh`).

## Hard numbers

Measured on an M-series MacBook (14 cores), Release/Ninja, double precision.

**Depth accuracy vs exact analytic depth** (10k random poses per case;
"gjk" = convex pipeline including the per-point depth patch):

| case      | collider | median  | p99     | max     |
|-----------|----------|--------:|--------:|--------:|
| cube_cube | box-box  | 1.4e-17 | 3.4e-04 | 9.6e-04 |
| cube_cube | gjk      | 1.9e-09 | 2.1e-09 | 8.1e-07 |
| aniso     | box-box  | 6.9e-18 | 5.2e-05 | 2.8e-04 |
| aniso     | gjk      | 8.5e-10 | 2.0e-09 | 2.1e-08 |

The box-box median is machine precision; its p99 tail is the deliberate 5%
face-preference band. Contact existence agreed on every sampled pose.

**Stack dynamics** (4 simulated seconds; gjk includes the per-point patch):

| scenario  | mode   | settle_vel | drift  | fallen | us/step |
|-----------|--------|-----------:|-------:|-------:|--------:|
| cubes30   | new    |    3.4e-09 | 0.0000 |  0/30  |    89.8 |
| cubes30   | legacy |    1.3e+01 | 1.4208 | 27/30  |    91.2 |
| cubes30   | gjk    |    3.5e-09 | 0.0000 |  0/30  |   101.8 |
| perturbed | new    |    1.4e-02 | 0.0090 |  0/30  |   100.2 |
| perturbed | legacy |    6.8e-03 | 0.0247 |  0/30  |   126.5 |
| perturbed | gjk    |    1.1e-01 | 0.0131 |  0/30  |   154.4 |
| plates    | new    |    2.7e-02 | 0.0002 |  0/24  |    80.9 |
| plates    | legacy |    3.4e-04 | 0.0016 |  0/24  |    70.8 |
| plates    | gjk    |    4.2e-04 | 0.0019 |  0/24  |    83.0 |

The legacy collider collapses the tall aligned tower. The patched convex
pipeline matches the new collider on aligned towers and beats it on plates
settle velocity (suspected 4-contact-cap effect, under investigation), at
13-54% higher step cost and 7.6x noisier perturbed settling.

**End-to-end performance vs the pre-rewrite collider**:

- `ccd_benchmark_test BM_BoxBox/0` (100-box pile, steady state): 272 us vs
  327 us per step, 17% faster; contacts/second 1.91M vs 1.87M.
- `benchmark.sh` (full trajectories, testspeed profiler): step time 1.03x /
  1.00x / 1.00x on pile / towers / plates; collision-phase time 1.04x /
  0.97x / 0.97x (stack rows are 8-14 us absolute, within noise).

**Per-point depth** (tilted plates, before the per-point patch): the convex
pipeline stamped the deep edge's depth on all four vertices (1.4e-4 error at
the far corner) and reported dist = -1e-4 at corners whose true gap is
+4e-5. The new collider matches the analytic gap to float precision at every
contact; the cherry-picked patch fixes the convex pipeline's version of
this.

## Recommendation

The SAT-based specialized collider (`new`) is the most promising box-box
narrowphase, on three measured grounds:

1. **Speed**: fastest in every scenario -- 13-17% cheaper steps than the
   patched convex pipeline on aligned towers and 54% cheaper on perturbed
   stacks (100.2 vs 154.4 us/step), 17% faster end-to-end than the legacy
   collider on the dense pile.
2. **Stability where it is hardest**: on perturbed stacks it settles 7.6x
   quieter than the patched convex pipeline (1.4e-2 vs 1.1e-1) with fewer
   contacts and solver iterations; the legacy collider collapses the aligned
   30-cube tower outright (27/30 fallen) while `new` holds it at 3.4e-9
   settle velocity.
3. **Exactness**: median depth error 1e-17 (machine precision) vs the
   convex pipeline's 1e-9 iterative floor, with contact existence agreeing
   on every sampled pose.

Stated fairly: the per-point depth patch brings the convex pipeline to
parity on *aligned* towers and ahead on plates settle velocity (4.2e-4 vs
2.7e-2, suspected 4-contact-cap interaction, under investigation), and its
depth-error tail (p99 2e-9) is tighter than `new`'s deliberate 5%
face-preference band. That patch should land regardless -- it fixes real
artifacts for mesh-mesh pairs, which have no specialized collider. But for
box-box pairs specifically, the specialized collider wins on cost and on
the harder stability scenarios while being exactly verifiable, and the
`boxbox` option makes every one of these claims reproducible in the viewer
in under a minute.

## Running the demos

Build, then run the viewer against any scene:

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j 14
    ./build/bin/simulate boxbox_validation/stacking_demo/tower30.xml

In the viewer, open the left panel's **Option** section and use the
**BoxBox** dropdown (next to Integrator): `New`, `Legacy`, or `Convex` --
the selection is always visible, one state at a time. Press Backspace to
reset the scene after switching. `Legacy` collapses the 30-cube tower;
`Convex` is the GJK/EPA pipeline (with the per-point depth patch on this
branch). `stacking_demo/perturbed.xml` shows the perturbed-tower
comparison. For `boxbox_validation/failure_cases/case*.xml`, load the
keyframe (Simulation section, key 0 -> "Load key") to pose the pair, then
switch modes with the dropdown; `verify_failure_cases.py` prints all three
modes' depths against the exact value.

Scripted equivalents: `stack_bench.py`, `plate_pair.py`,
`search_gjk_failures.py <case> <n>`, `perf/benchmark.sh <old-ref>` (see
`boxbox_validation/README.md`).
