"""Types."""
import taichi as ti
import mujoco


@ti.data_oriented
class Model:
  def __init__(self, m: mujoco.MjModel, nlevel: int):
    self.nq = m.nq
    self.nv = m.nv
    self.nbody = m.nbody
    self.njnt = m.njnt
    self.ngeom = m.ngeom
    self.nsite = m.nsite
    self.nmocap = m.nmocap
    self.nlevel = nlevel
    self.qpos0 = ti.field(dtype=ti.f32, shape=self.nq)
    self.level_beg = ti.field(dtype=ti.i32, shape=self.nlevel)
    self.level_end = ti.field(dtype=ti.i32, shape=self.nlevel)
    self.body_bfs = ti.field(dtype=ti.i32, shape=self.nbody - 1)
    self.body_jntadr = ti.field(dtype=ti.i32, shape=self.nbody)
    self.body_jntnum = ti.field(dtype=ti.i32, shape=self.nbody)
    self.body_parentid = ti.field(dtype=ti.i32, shape=self.nbody)
    self.body_mocapid = ti.field(dtype=ti.i32, shape=self.nbody)
    self.body_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.nbody)
    self.body_quat = ti.Vector.field(4, dtype=ti.f32, shape=self.nbody)
    self.body_ipos = ti.Vector.field(3, dtype=ti.f32, shape=self.nbody)
    self.body_iquat = ti.Vector.field(4, dtype=ti.f32, shape=self.nbody)
    self.jnt_type = ti.field(dtype=ti.i32, shape=self.njnt)
    self.jnt_qposadr = ti.field(dtype=ti.i32, shape=self.njnt)
    self.jnt_axis = ti.Vector.field(3, dtype=ti.f32, shape=self.njnt)
    self.jnt_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.njnt)
    self.geom_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.ngeom)
    self.geom_quat = ti.Vector.field(4, dtype=ti.f32, shape=self.ngeom)
    if self.nsite:
      self.site_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.nsite)
      self.site_quat = ti.Vector.field(4, dtype=ti.f32, shape=self.nsite)


@ti.data_oriented
class Data:
  def __init__(self, m: mujoco.MjModel, nworld: int):
    self.nworld = nworld
    self.qpos = ti.field(dtype=ti.f32, shape=(nworld, m.nq))
    if m.nmocap:
      self.mocap_pos = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.nmocap))
      self.mocap_quat = ti.Vector.field(4, dtype=ti.f32, shape=(nworld, m.nmocap))
    self.xanchor = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.njnt))
    self.xaxis = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.njnt))
    self.xmat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(nworld, m.nbody))
    self.xpos = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.nbody))
    self.xquat = ti.Vector.field(4, dtype=ti.f32, shape=(nworld, m.nbody))
    self.xipos = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.nbody))
    self.ximat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(nworld, m.nbody))
    self.geom_xpos = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.ngeom))
    self.geom_xmat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(nworld, m.ngeom))
    if m.nsite:
      self.site_xpos = ti.Vector.field(3, dtype=ti.f32, shape=(nworld, m.nsite))
      self.site_xmat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(nworld, m.nsite))
