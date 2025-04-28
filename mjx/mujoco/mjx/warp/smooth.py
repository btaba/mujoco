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
import inspect
import functools
import jax
import jax.numpy as jp
import warp as wp
from warp.jax_experimental import ffi as warp_ffi
import mujoco_warp as mjwarp

from mujoco.mjx._src import types


# def _reshape_batch_dim(*args):
#   # pytype: disable=attribute-error
#   new_args = [None] * len(args)
#   for i in range(len(args)):
#     expected_ndim = mjwarp.kinematics_.adj[i].type.ndim
#     new_args[i] = args[i].reshape((-1,) + args[i].shape[:expected_ndim])
#     new_args[i].ndim = expected_ndim
#   # pytype: enable=attribute-error
#   return new_args


# def _decorator(wrapped_func):
#   sig = inspect.signature(wrapped_func)

#   @functools.wraps(wrapped_func)
#   def wrapper(*args):
#     bound_args = sig.bind(*args)
#     new_args_tuple = _reshape_batch_dim(bound_args)
#     return wrapped_func(*new_args_tuple)
#   return wrapper


def _kinematics_shim(
    # Model Inputs
    body_tree: wp.array(dtype=int),
    qpos0: wp.array(dtype=float),
    body_parentid: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_pos: wp.array(dtype=wp.vec3),
    body_quat: wp.array(dtype=wp.quat),
    body_ipos: wp.array(dtype=wp.vec3),
    body_iquat: wp.array(dtype=wp.quat),
    body_treeadr: wp.array(dtype=int),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_pos: wp.array(dtype=wp.vec3),
    jnt_axis: wp.array(dtype=wp.vec3),
    geom_bodyid: wp.array(dtype=int),
    # TODO(btaba): test batching on `geom_pos` as an example
    geom_pos: wp.array(dtype=wp.vec3),
    geom_quat: wp.array(dtype=wp.quat),
    site_bodyid: wp.array(dtype=int),
    site_pos: wp.array(dtype=wp.vec3),
    site_quat: wp.array(dtype=wp.quat),
    # Data Inputs
    qpos: wp.array1d(dtype=float),  # TODO(btaba): Parameter dims are misleading with vmap, should be fixed in Warp.
    # Data Outputs
    xpos: wp.array1d(dtype=wp.vec3),
    xquat: wp.array1d(dtype=wp.quat),
    xmat: wp.array1d(dtype=wp.mat33),
    xipos: wp.array1d(dtype=wp.vec3),
    ximat: wp.array1d(dtype=wp.mat33),
    xanchor: wp.array1d(dtype=wp.vec3),
    xaxis: wp.array1d(dtype=wp.vec3),
    geom_xpos: wp.array1d(dtype=wp.vec3),
    geom_xmat: wp.array1d(dtype=wp.mat33),
    site_xpos: wp.array1d(dtype=wp.vec3),
    site_xmat: wp.array1d(dtype=wp.mat33),
):
  args = (
      body_tree,
      qpos0,
      body_parentid,
      body_jntadr,
      body_jntnum,
      body_pos,
      body_quat,
      body_ipos,
      body_iquat,
      body_treeadr,
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
      qpos,
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
  # pytype: disable=attribute-error
  new_args = [None] * len(args)
  annotations = tuple(mjwarp.kinematics_.__annotations__.items())
  for i in range(len(args)):
    expected_ndim = annotations[i][1].ndim
    if expected_ndim < args[i].ndim:
      # remove the expanded dims, this assumes we used `expand_dims` vmap_method
      assert args[i].shape[0] == 1
      new_args[i] = args[i].reshape(args[i].shape[args[i].ndim - expected_ndim:])
      new_args[i].ndim = args[i].ndim - 1
      continue

    new_args[i] = args[i]
  # pytype: enable=attribute-error
  mjwarp.kinematics_(*new_args)


# wp.config.verbose = True
# wp.config.print_launches = True
# wp.config.mode = "debug"
# wp.config.verify_cuda = True


def kinematics(m: types.Model, d: types.Data) -> types.Data:
  """Forward kinematics."""
  output_dims = {
      'xpos': (m.nbody, 3),
      'xquat': (m.nbody, 4),
      'xmat': (m.nbody, 3, 3),
      'xipos': (m.nbody, 3),
      'ximat': (m.nbody, 3, 3),
      'xanchor': (m.njnt, 3),
      'xaxis': (m.njnt, 3),
      'geom_xpos': (m.ngeom, 3),
      'geom_xmat': (m.ngeom, 3, 3),
      'site_xpos': (m.nsite, 3),
      'site_xmat': (m.nsite, 3, 3),
  }
  jf = warp_ffi.jax_callable(
      _kinematics_shim, num_outputs=11,
      output_dims=output_dims,
      vmap_method='expand_dims',
      graph_compatible=True,
    )

  out = jf(
      m.body_tree,
      m.qpos0,
      m.body_parentid,
      m.body_jntadr,
      m.body_jntnum,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.body_treeadr,
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
      d.qpos,
  )

  d = d.replace(
      xpos=out[0],
      xquat=out[1],
      xmat=out[2],
      xipos=out[3],
      ximat=out[4],
      xanchor=out[5],
      xaxis=out[6],
      geom_xpos=out[7],
      geom_xmat=out[8],
      site_xpos=out[9],
      site_xmat=out[10],
  )
  return d
