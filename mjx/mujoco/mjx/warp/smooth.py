# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""mjWarp Smooth."""
import dataclasses
from mujoco.mjx._src import types
import mujoco_warp as mjwarp
import warp as wp
from warp.jax_experimental import ffi as warp_ffi
import numpy as np


def _kinematics(
    # Model
    ngeom: int,
    nsite: int,
    nmocap: int,
    qpos0: wp.array(dtype=float),
    # body_tree: tuple[wp.array(dtype=int), ...],
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

  body_tree = []
  body_tree_adr = body_tree_adr.numpy()
  body_tree_val = body_tree_val.numpy()
  for i in range(len(body_tree_adr) - 1):
    beg = body_tree_adr[i]
    end = body_tree_adr[i + 1]
    body_tree.append(wp.array(body_tree_val[beg:end]))
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
    # body_tree: tuple[wp.array(dtype=int), ...],
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
      # body_tree,
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
      # "body_tree",
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
  new_args = [None] * len(args)
  annotations = tuple(_kinematics.__annotations__.items())
  for i in range(len(args)):
    typ_ = annotations[i][1]
    if not hasattr(typ_, 'ndim'):
      new_args[i] = args[i]
      continue
    expected_ndim = annotations[i][1].ndim

    # Remove the expanded_dim if we are exceeding the expected ndim.
    # i.e. Unbatched model fields will get an extra dim due to
    # vmap_method="expand_dims".
    if args[i].ndim > expected_ndim and args[i].shape[0] == 1:
      extra_dim = args[i].ndim - expected_ndim
      assert sum(args[i].shape[:extra_dim]) == extra_dim
      arg = args[i].reshape(args[i].shape[extra_dim:])
      arg.ndim = expected_ndim
      new_args[i] = arg
      print(
          "Removing extra dim: ",
          names[i],
          args[i].shape,
          "=>",
          new_args[i].shape,
      )
      continue

    # Squash nested vmap batch axes.
    if args[i].ndim > expected_ndim:
      extra_ndim = args[i].ndim - expected_ndim
      new_args[i] = args[i].reshape((-1,) + args[i].shape[extra_ndim + 1 :])
      new_args[i].ndim = expected_ndim
      print(
          "Squashing extra dim: ",
          names[i],
          args[i].shape,
          "=>",
          new_args[i].shape,
      )
      continue

    # if args[i].ndim > expected_ndim:
    #   print(args[i], args[i].shape, args[i].ndim)
    #   raise ValueError(f'arg[{i}] has ndim={args[i].ndim}, expected ndim={expected_ndim}.')

    # Add stride 0 to unbatched inputs that have the correct ndim.
    # Unbatched inputs get a leading dimension of 1, using
    # vmap_method="expand_dims". We add a stride of 0 to the leading dim.
    if expected_ndim == args[i].ndim and args[i].shape[0] == 1:
      arg = args[i]
      old_strides = arg.strides
      arg.strides = (0,) + arg.strides[1:]
      new_args[i] = arg
      print(
          "Leading batch dim of 1, adding stride: ",
          names[i],
          old_strides,
          "=>",
          new_args[i].strides,
      )
      continue

    # Add batch dims if they don't exist. This occurs when the underlying
    # function expects a batch dim but the outer function was not called
    # with vmap.
    # e.g. Model/Data have nworld == 1, but without the leading dim in JAX.
    if expected_ndim > args[i].ndim:
      extra_dims = expected_ndim - args[i].ndim
      new_args[i] = args[i].reshape((1,) * extra_dims + args[i].shape)
      new_args[i].ndim = expected_ndim
      print(
          "No leading batch dims",
          names[i],
          args[i].shape,
          "=>",
          new_args[i].shape,
      )
      continue

    new_args[i] = args[i]
    print("Did nothing: ", names[i], args[i].shape, "=>", new_args[i].shape)

  _kinematics(*new_args)


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
  jf = warp_ffi.jax_callable(
      _kinematics_shim,
      num_outputs=11,
      output_dims=output_dims,
      vmap_method="expand_dims",
      graph_compatible=True,
  )

  body_tree_val = np.concatenate(m.body_tree, dtype=np.int32)
  body_tree_adr = np.cumsum([len(v) for v in m.body_tree], dtype=np.int32)
  body_tree_adr = np.append(np.array(0), body_tree_adr).astype(np.int32)

  out = jf(
      m.ngeom,
      m.nsite,
      m.nmocap,
      m.qpos0,
      # m.body_tree,
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
