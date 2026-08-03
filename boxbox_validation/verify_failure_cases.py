# Verifies the accuracy-case models: for each keyframed pose, prints the
# exact penetration depth and each collider mode's deepest contact
# (mjOption.boxbox: new / legacy / convex).
import glob
import os

import mujoco
import numpy as np


def brute_force_sep(size1, size2, pos1, mat1, pos2, mat2):
  """Exact box-box separation: for penetration the optimum is one of the 15
  candidate axes (6 face normals, 9 edge cross products), evaluated here
  independently of the engine in float64."""
  R1 = np.asarray(mat1).reshape(3, 3)
  R2 = np.asarray(mat2).reshape(3, 3)
  axes = [R1[:, i] for i in range(3)] + [R2[:, j] for j in range(3)]
  for i in range(3):
    for j in range(3):
      c = np.cross(R1[:, i], R2[:, j])
      n = np.linalg.norm(c)
      if n > 1e-8:
        axes.append(c / n)
  dpos = np.asarray(pos2) - np.asarray(pos1)
  best = -np.inf
  for a in axes:
    r1 = np.abs(a @ R1) @ size1
    r2 = np.abs(a @ R2) @ size2
    best = max(best, abs(a @ dpos) - r1 - r2)
  return float(best)


def deepest(data, gset):
  d = None
  for i in range(data.ncon):
    c = data.contact[i]
    if {c.geom1, c.geom2} == gset:
      d = c.dist if d is None else min(d, c.dist)
  return d


MODES = {"new": 0, "legacy": 1, "convex": 2}


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  paths = sorted(glob.glob(os.path.join(here, "failure_cases", "case*.xml")))
  hdr = "".join(f"{m:>13}{m + ' err':>13}" for m in MODES)
  print(f"{'case':<10}{'exact depth':>13}{hdr}")
  for path in paths:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    true_sep = brute_force_sep(
        model.geom_size[0], model.geom_size[1],
        data.geom_xpos[0], data.geom_xmat[0],
        data.geom_xpos[1], data.geom_xmat[1])
    row = f"{os.path.basename(path):<10}{true_sep:>13.6f}"
    for mode, val in MODES.items():
      model.opt.boxbox = val
      mujoco.mj_resetDataKeyframe(model, data, 0)
      mujoco.mj_forward(model, data)
      db = deepest(data, {0, 1})
      if db is None:
        row += f"{'---':>13}{'---':>13}"
      else:
        row += f"{db:>13.6f}{abs(db - true_sep):>13.2e}"
    print(row)
  print("\nerr columns: absolute deviation of each mode's deepest contact "
        "from the exact depth.")


if __name__ == "__main__":
  main()
