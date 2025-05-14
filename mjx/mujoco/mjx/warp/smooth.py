import dataclasses
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi_helper
import mujoco_warp as mjwarp
import warp as wp
from warp.jax_experimental import ffi as warp_ffi


def _kinematics(
    # Model
    ngeom: int,
    nsite: int,
    nmocap: int,
    qpos0: wp.array2d(dtype=float),
    body_tree_val: wp.array(dtype=int),
    body_tree_adr: wp.array(dtype=int),
    body_parentid: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    geom_bodyid: wp.array(dtype=int),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    site_bodyid: wp.array(dtype=int),
    site_pos: wp.array2d(dtype=wp.vec3),
    site_quat: wp.array2d(dtype=wp.quat),
    mocap_bodyid: wp.array(dtype=int),
    # Data
    qpos: wp.array2d(dtype=float),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    xmat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3),
    ximat: wp.array2d(dtype=wp.mat33),
    xanchor: wp.array2d(dtype=wp.vec3),
    xaxis: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
):

  m = mjwarp.Model(
      **{f.name: None for f in dataclasses.fields(mjwarp.Model) if f.init}
  )
  d = mjwarp.Data(
      **{f.name: None for f in dataclasses.fields(mjwarp.Data) if f.init}
  )

  m.body_ipos = body_ipos
  m.body_iquat = body_iquat
  m.body_jntadr = body_jntadr
  m.body_jntnum = body_jntnum
  m.body_parentid = body_parentid
  m.body_pos = body_pos
  m.body_quat = body_quat
  body_tree = ffi_helper.adr_arr_to_tuple(body_tree_val, body_tree_adr)
  m.body_tree = body_tree
  m.geom_bodyid = geom_bodyid
  m.geom_pos = geom_pos
  m.geom_quat = geom_quat
  m.jnt_axis = jnt_axis
  m.jnt_pos = jnt_pos
  m.jnt_qposadr = jnt_qposadr
  m.jnt_type = jnt_type
  m.mocap_bodyid = mocap_bodyid
  m.ngeom = ngeom
  m.nmocap = nmocap
  m.nsite = nsite
  m.qpos0 = qpos0
  m.site_bodyid = site_bodyid
  m.site_pos = site_pos
  m.site_quat = site_quat
  d.geom_xmat = geom_xmat
  d.geom_xpos = geom_xpos
  d.mocap_pos = mocap_pos
  d.mocap_quat = mocap_quat
  d.qpos = qpos
  d.site_xmat = site_xmat
  d.site_xpos = site_xpos
  d.xanchor = xanchor
  d.xaxis = xaxis
  d.ximat = ximat
  d.xipos = xipos
  d.xmat = xmat
  d.xpos = xpos
  d.xquat = xquat
  d.nworld = d.qpos.shape[0]
  mjwarp.kinematics(m, d)


def _kinematics_shim(
    ngeom: int,
    nsite: int,
    nmocap: int,
    qpos0: wp.array(dtype=float),
    body_tree_val: wp.array(dtype=int),
    body_tree_adr: wp.array(dtype=int),
    body_parentid: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_pos: wp.array(dtype=wp.vec3),
    body_quat: wp.array(dtype=wp.quat),
    body_ipos: wp.array(dtype=wp.vec3),
    body_iquat: wp.array(dtype=wp.quat),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_pos: wp.array(dtype=wp.vec3),
    jnt_axis: wp.array(dtype=wp.vec3),
    geom_bodyid: wp.array(dtype=int),
    geom_pos: wp.array(dtype=wp.vec3),
    geom_quat: wp.array(dtype=wp.quat),
    site_bodyid: wp.array(dtype=int),
    site_pos: wp.array(dtype=wp.vec3),
    site_quat: wp.array(dtype=wp.quat),
    mocap_bodyid: wp.array(dtype=int),
    qpos: wp.array(dtype=float),
    mocap_pos: wp.array(dtype=wp.vec3),
    mocap_quat: wp.array(dtype=wp.quat),
    xpos: wp.array(dtype=wp.vec3),
    xquat: wp.array(dtype=wp.quat),
    xmat: wp.array(dtype=wp.mat33),
    xipos: wp.array(dtype=wp.vec3),
    ximat: wp.array(dtype=wp.mat33),
    xanchor: wp.array(dtype=wp.vec3),
    xaxis: wp.array(dtype=wp.vec3),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    site_xpos: wp.array(dtype=wp.vec3),
    site_xmat: wp.array(dtype=wp.mat33),
):
  args = (
      ngeom,
      nsite,
      nmocap,
      qpos0,
      body_tree_val,
      body_tree_adr,
      body_parentid,
      body_jntnum,
      body_jntadr,
      body_pos,
      body_quat,
      body_ipos,
      body_iquat,
      jnt_type,
      jnt_qposadr,
      jnt_pos,
      jnt_axis,
      geom_bodyid,
      geom_pos,
      geom_quat,
      site_bodyid,
      site_pos,
      site_quat,
      mocap_bodyid,
      qpos,
      mocap_pos,
      mocap_quat,
      xpos,
      xquat,
      xmat,
      xipos,
      ximat,
      xanchor,
      xaxis,
      geom_xpos,
      geom_xmat,
      site_xpos,
      site_xmat,
  )
  names = (
      "ngeom",
      "nsite",
      "nmocap",
      "qpos0",
      "body_tree_val",
      "body_tree_adr",
      "body_parentid",
      "body_jntnum",
      "body_jntadr",
      "body_pos",
      "body_quat",
      "body_ipos",
      "body_iquat",
      "jnt_type",
      "jnt_qposadr",
      "jnt_pos",
      "jnt_axis",
      "geom_bodyid",
      "geom_pos",
      "geom_quat",
      "site_bodyid",
      "site_pos",
      "site_quat",
      "mocap_bodyid",
      "qpos",
      "mocap_pos",
      "mocap_quat",
      "xpos",
      "xquat",
      "xmat",
      "xipos",
      "ximat",
      "xanchor",
      "xaxis",
      "geom_xpos",
      "geom_xmat",
      "site_xpos",
      "site_xmat",
  )
  args = ffi_helper.format_args_for_warp(*args, names=names, kernel=_kinematics)
  _kinematics(*args)


def kinematics(m: types.Model, d: types.Data):
  output_dims = {
      "xpos": (m.nbody, 3),
      "xquat": (m.nbody, 4),
      "xmat": (m.nbody, 3, 3),
      "xipos": (m.nbody, 3),
      "ximat": (m.nbody, 3, 3),
      "xanchor": (m.njnt, 3),
      "xaxis": (m.njnt, 3),
      "geom_xpos": (m.ngeom, 3),
      "geom_xmat": (m.ngeom, 3, 3),
      "site_xpos": (m.nsite, 3),
      "site_xmat": (m.nsite, 3, 3),
  }
  body_tree_val, body_tree_adr = ffi_helper.tuple_to_adr_arr(m.body_tree)
  jf = warp_ffi.jax_callable(
      _kinematics_shim,
      num_outputs=11,
      output_dims=output_dims,
      vmap_method="expand_dims",
      graph_compatible=True,
  )
  out = jf(
      m.ngeom,
      m.nsite,
      m.nmocap,
      m.qpos0,
      body_tree_val,
      body_tree_adr,
      m.body_parentid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_pos,
      m.jnt_axis,
      m.geom_bodyid,
      m.geom_pos,
      m.geom_quat,
      m.site_bodyid,
      m.site_pos,
      m.site_quat,
      m.mocap_bodyid,
      d.qpos,
      d.mocap_pos,
      d.mocap_quat,
  )
  d = d.tree_replace({
      "xpos": out[0],
      "xquat": out[1],
      "xmat": out[2],
      "xipos": out[3],
      "ximat": out[4],
      "xanchor": out[5],
      "xaxis": out[6],
      "geom_xpos": out[7],
      "geom_xmat": out[8],
      "site_xpos": out[9],
      "site_xmat": out[10],
  })
  return d
