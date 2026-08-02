# Searches for poses where GJK/EPA on a box-mesh pair misreports penetration
# while the box-box collider matches ground truth. Poses are applied through
# qpos and mj_kinematics, so the mesh's compiled principal-axis frame is
# honored (overwriting geom_xmat directly would silently permute a thin
# mesh's axes).
import sys

import mujoco
import numpy as np

SIZES = {
    "slab_slab": ((0.08, 0.08, 0.004), (0.06, 0.06, 0.006)),
    "aniso": ((0.018, 0.038, 0.047), (0.026, 0.0014, 0.008)),
    "slab_needle": ((0.07, 0.05, 0.003), (0.0015, 0.0015, 0.06)),
    "cube_cube": ((0.05, 0.05, 0.05), (0.05, 0.05, 0.05)),
}

XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <asset>
    <mesh name="m1" scale="{s1x} {s1y} {s1z}"
      vertex="-1 -1 -1  1 -1 -1  1 1 -1  1 1 1  1 -1 1  -1 1 -1  -1 1 1  -1 -1 1"/>
    <mesh name="m2" scale="{s2x} {s2y} {s2z}"
      vertex="-1 -1 -1  1 -1 -1  1 1 -1  1 1 1  1 -1 1  -1 1 -1  -1 1 1  -1 -1 1"/>
  </asset>
  <worldbody>
    <body><freejoint/>
      <geom type="box" size="{s1x} {s1y} {s1z}" contype="1" conaffinity="1"/></body>
    <body><freejoint/>
      <geom type="box" size="{s2x} {s2y} {s2z}" contype="1" conaffinity="1"/></body>
    <body pos="0 5 0"><freejoint/>
      <geom type="mesh" mesh="m1" contype="2" conaffinity="2"/></body>
    <body pos="0 5 0"><freejoint/>
      <geom type="mesh" mesh="m2" contype="2" conaffinity="2"/></body>
  </worldbody>
</mujoco>
"""

def true_sep(size1, size2, pos1, mat1, pos2, mat2):
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
  case = sys.argv[1] if len(sys.argv) > 1 else "slab_slab"
  n_poses = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
  s1, s2 = SIZES[case]
  model = mujoco.MjModel.from_xml_string(XML.format(
      s1x=s1[0], s1y=s1[1], s1z=s1[2], s2x=s2[0], s2y=s2[1], s2z=s2[2]))
  data = mujoco.MjData(model)
  scale = max(s1) + max(s2)

  rng = np.random.RandomState(0)
  worst = []
  berrs, gerrs = [], []
  stats = {"both": 0, "only_box": 0, "only_gjk": 0, "n": 0}
  for it in range(n_poses):
    pos2 = 1.15 * scale * rng.uniform(-1, 1, 3)
    q1 = rng.normal(size=4); q1 /= np.linalg.norm(q1)
    q2 = rng.normal(size=4); q2 /= np.linalg.norm(q2)
    qpos = np.concatenate([
        [0, 0, 0], q1, pos2, q2,
        [0, 5, 0], q1, pos2 + [0, 5, 0], q2])
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    db = deepest(data, {0, 1})
    dg = deepest(data, {2, 3})
    stats["n"] += 1
    if db is not None and dg is not None:
      stats["both"] += 1
      if db < 0 and dg < 0:
        ts = true_sep(np.array(s1), np.array(s2), data.geom_xpos[0],
                      data.geom_xmat[0], data.geom_xpos[1], data.geom_xmat[1])
        berr, gerr = abs(db - ts), abs(dg - ts)
        berrs.append(berr)
        gerrs.append(gerr)
        if gerr > 10 * max(berr, 1e-12):
          worst.append((gerr, berr, ts, db, dg, pos2.copy(), q1.copy(),
                        q2.copy()))
    elif db is not None:
      stats["only_box"] += 1
    elif dg is not None:
      stats["only_gjk"] += 1

  print(f"[{case}] {stats}")
  b, g = np.array(berrs), np.array(gerrs)
  for name, e in (("boxbox", b), ("gjk   ", g)):
    print(f"  {name} err vs exact depth: median={np.median(e):.2e} "
          f"p90={np.percentile(e, 90):.2e} p99={np.percentile(e, 99):.2e} "
          f"max={e.max():.2e}")
  worst.sort(reverse=True)
  print(f"  poses where gjk err > 10x boxbox err: {len(worst)} "
        f"({100.0 * len(worst) / max(len(b), 1):.1f}% of contacts)")
  for gerr, berr, ts, db, dg, pos2, q1, q2 in worst[:4]:
    print(f"  gjk_err={gerr:.3e} box_err={berr:.3e} true={ts:+.6f} "
          f"db={db:+.6f} dg={dg:+.6f}")
    print(f"    pos2={list(pos2)}")
    print(f"    quat1={list(q1)}")
    print(f"    quat2={list(q2)}")


if __name__ == "__main__":
  main()
