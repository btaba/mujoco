import taichi as ti
import taichi.math as tm
import mujoco
import numpy as np

from . import types


def put_model(m: mujoco.MjModel) -> types.Model:
  # body_bfs is BFS ordering of body ids
  # level_beg, level_end specify the bounds of level ranges in body_range
  level_beg, level_end, body_bfs = [], [], []
  parents = {0}
  while len(body_bfs) < m.nbody - 1:
    children = [i for i, p in enumerate(m.body_parentid) if p in parents and i != 0]
    if not children:
      raise ValueError("invalid tree layout")
    level_beg.append(len(body_bfs))
    body_bfs.extend(children)
    level_end.append(len(body_bfs))
    parents = set(children)

  nlevel = len(level_beg)
  mx = types.Model(m, nlevel)

  # static fields
  mx.level_beg = ti.Vector(np.array(level_beg), ti.i32)
  mx.level_end = ti.Vector(np.array(level_end), ti.i32)

  # dynamic
  mx.body_bfs.from_numpy(np.array(body_bfs))
  mx.body_jntadr.from_numpy(m.body_jntadr)
  mx.body_jntnum.from_numpy(m.body_jntnum)
  mx.body_parentid.from_numpy(m.body_parentid)
  mx.body_mocapid.from_numpy(m.body_mocapid)
  mx.jnt_type.from_numpy(m.jnt_type)
  mx.jnt_qposadr.from_numpy(m.jnt_qposadr)

  # Allocate and assign "non-static" fields that may slow down compilation.
  # This also gets rid of taichi warnings about ti.Vectors larger than
  #  32 entries possibly slowing down compilation.
  mx.qpos0.from_numpy(m.qpos0)
  mx.body_pos.from_numpy(m.body_pos)
  mx.body_quat.from_numpy(m.body_quat)
  mx.body_ipos.from_numpy(m.body_ipos)
  mx.body_iquat.from_numpy(m.body_iquat)
  mx.jnt_axis.from_numpy(m.jnt_axis)
  mx.jnt_pos.from_numpy(m.jnt_pos)
  mx.geom_pos.from_numpy(m.geom_pos)
  mx.geom_quat.from_numpy(m.geom_quat)
  if mx.nsite:
    mx.site_pos.from_numpy(m.site_pos)
    mx.site_quat.from_numpy(m.site_quat)

  return mx


def make_data(m: mujoco.MjModel, nworld: int = 1) -> types.Data:
  d = types.Data(m, nworld)
  d.nworld = nworld
  qpos0 = np.tile(m.qpos0, (nworld, 1))
  d.qpos.from_numpy(qpos0)
  return d
