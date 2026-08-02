# Verifies the failure-case models: for each keyframed pose, prints the true
# penetration depth (dense direction sweep over support separations, exact for
# boxes in the limit), the box-box collider's deepest contact, and GJK/EPA's
# deepest contact on the identical mesh pair.
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


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  paths = sorted(glob.glob(os.path.join(here, "failure_cases", "case*.xml")))
  print(f"{'case':<10}{'true depth':>14}{'boxbox':>14}{'gjk/epa':>14}"
        f"{'boxbox err':>13}{'gjk err':>13}")
  for path in paths:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    true_sep = brute_force_sep(
        model.geom_size[0], model.geom_size[1],
        data.geom_xpos[0], data.geom_xmat[0],
        data.geom_xpos[1], data.geom_xmat[1])
    db = deepest(data, {0, 1})
    dg = deepest(data, {2, 3})
    name = os.path.basename(path)
    if db is None or dg is None:
      print(f"{name:<10}{true_sep:>14.6f}{str(db):>14}{str(dg):>14}"
            f"  (missing contact)")
      continue
    print(f"{name:<10}{true_sep:>14.6f}{db:>14.6f}{dg:>14.6f}"
          f"{abs(db - true_sep):>13.2e}{abs(dg - true_sep):>13.2e}")
  print("\nerr columns: absolute deviation of each method's deepest contact "
        "from the true depth.")


if __name__ == "__main__":
  main()
