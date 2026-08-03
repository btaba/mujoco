# Dynamics benchmark comparing the three box-box collider modes on stacking
# scenarios, mirroring the metrics of the ccd-manifold-depth proposal's
# stack_bench (settle velocity, drift, fallen count, contact count, solver
# iterations, microseconds per step).
#
# Modes are selected with the collider flags:
#   new    -- default (specialized box-box collider)
#   legacy -- <flag boxboxlegacy="enable"/>
#   gjk    -- <flag boxbox="disable"/> (general convex pipeline, GJK/EPA)
#
# Usage: python stack_bench.py [sim_seconds]
import sys
import time

import mujoco
import numpy as np


def tower_xml(n, size, jitter_deg, jitter_xy, nrow=1):
  rng = np.random.RandomState(12345)
  bodies = []
  for r in range(nrow):
    x0, y0 = 0.5 * r, 0.0
    for i in range(n):
      z = (2 * i + 1) * size[2] * 1.001
      dx, dy = rng.uniform(-jitter_xy, jitter_xy, 2)
      yaw = rng.uniform(-jitter_deg, jitter_deg)
      bodies.append(
          f'<body pos="{x0 + dx} {y0 + dy} {z}" euler="0 0 {yaw}">'
          f'<freejoint/><geom type="box" size="{size[0]} {size[1]} {size[2]}"/>'
          f'</body>')
  return f"""
<mujoco>
  <option timestep="2e-3">
    <flag {{flag}}/>
  </option>
  <worldbody>
    <geom type="plane" size="10 10 .1"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""

SCENARIOS = {
    "cubes10":    tower_xml(10, (0.05, 0.05, 0.05), 0, 0),
    "cubes20":    tower_xml(20, (0.05, 0.05, 0.05), 0, 0),
    "cubes30":    tower_xml(30, (0.05, 0.05, 0.05), 0, 0),
    "perturbed":  tower_xml(10, (0.05, 0.05, 0.05), 5, 0.005, nrow=3),
    "plates":     tower_xml(8, (0.08, 0.08, 0.008), 3, 0.004, nrow=3),
}

MODES = {
    "new": 'contact="enable"',              # no-op attribute: defaults
    "legacy": 'boxboxlegacy="enable"',
    "gjk": 'boxbox="disable"',
}


def run(xml, sim_seconds):
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  nbody = model.nbody - 1
  z0 = np.array([data.qpos[7 * i + 2] for i in range(nbody)])
  xy0 = np.array([data.qpos[7 * i:7 * i + 2] for i in range(nbody)])

  nsteps = int(sim_seconds / model.opt.timestep)
  t0 = time.perf_counter()
  iters = 0
  for _ in range(nsteps):
    mujoco.mj_step(model, data)
    iters += data.solver_niter[0]
  us_per_step = 1e6 * (time.perf_counter() - t0) / nsteps

  settle_vel = float(np.abs(data.qvel).max())
  z1 = np.array([data.qpos[7 * i + 2] for i in range(nbody)])
  xy1 = np.array([data.qpos[7 * i:7 * i + 2] for i in range(nbody)])
  drift = float(np.linalg.norm(xy1 - xy0, axis=1).max())
  fallen = int(np.sum(z1 < 0.5 * z0))
  return dict(settle_vel=settle_vel, drift=drift, fallen=fallen,
              ncon=data.ncon, iters=iters // max(nsteps, 1),
              us=us_per_step)


def main():
  sim_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
  print(f"simulating {sim_seconds}s per scenario per mode\n")
  print(f"{'scenario':<11}{'mode':<8}{'settle_vel':>11}{'drift':>9}"
        f"{'fallen':>7}{'ncon':>6}{'iters':>6}{'us/step':>9}")
  for name, xml in SCENARIOS.items():
    for mode, flag in MODES.items():
      r = run(xml.format(flag=flag), sim_seconds)
      print(f"{name:<11}{mode:<8}{r['settle_vel']:>11.2e}{r['drift']:>9.4f}"
            f"{r['fallen']:>7}{r['ncon']:>6}{r['iters']:>6}{r['us']:>9.1f}")
    print()


if __name__ == "__main__":
  main()
