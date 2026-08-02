#!/bin/bash
# Benchmarks the box-box collider on box-dominated scenes, comparing the
# current checkout against a reference revision (default: the commit before
# the box-box rewrite on this branch).
#
# Usage: ./benchmark.sh [old-ref] [nstep]
#
# Builds the reference revision in a temporary git worktree, builds the
# current checkout, then runs testspeed on each scene with both binaries and
# reports microseconds per step (thread-0 internal profiler, "step" row).
set -euo pipefail

OLD_REF="${1:-$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD~1)}"
NSTEP="${2:-4000}"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SCENES=(pile.xml towers.xml plates.xml)

build() {
  local src="$1" bdir="$2"
  cmake -S "$src" -B "$bdir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
    -DMUJOCO_TEST_PYTHON_UTIL=OFF > /dev/null
  cmake --build "$bdir" --target testspeed -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" > /dev/null
}

run_scene() {
  # prints "<step_us> <collision_us>" from the thread-0 internal profiler
  "$1" "$2" --nstep="$NSTEP" 2>/dev/null | awk '
    /^ +step :/ {step=$3} /^ +collision +:/ {col=$3}
    END {print step, col}'
}

echo "reference revision: $OLD_REF"
WT="$(mktemp -d)/mujoco-old"
git -C "$REPO" worktree add --detach "$WT" "$OLD_REF" > /dev/null

trap 'git -C "$REPO" worktree remove --force "$WT" > /dev/null 2>&1 || true' EXIT

echo "building reference..."
build "$WT" "$WT/build_bench"
echo "building current checkout..."
build "$REPO" "$REPO/build/bench"

printf "\n%-13s %11s %11s %8s %13s %13s %8s\n" \
  "scene" "old step" "new step" "speedup" "old collide" "new collide" "speedup"
for scene in "${SCENES[@]}"; do
  read -r old_step old_col <<< "$(run_scene "$WT/build_bench/bin/testspeed" "$SCRIPT_DIR/$scene")"
  read -r new_step new_col <<< "$(run_scene "$REPO/build/bench/bin/testspeed" "$SCRIPT_DIR/$scene")"
  s1="$(echo "$old_step $new_step" | awk '{printf "%.2fx", $1 / $2}')"
  s2="$(echo "$old_col $new_col" | awk '{printf "%.2fx", $1 / $2}')"
  printf "%-13s %11s %11s %8s %13s %13s %8s\n" \
    "$scene" "$old_step" "$new_step" "$s1" "$old_col" "$new_col" "$s2"
done
echo
echo "times are us per step (testspeed thread-0 profiler); collide = collision row"
