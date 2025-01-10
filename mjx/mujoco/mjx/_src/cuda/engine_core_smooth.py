# Copyright 2024 DeepMind Technologies Limited
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
"""Core smooth cuda functions."""

import jax
from jax import numpy as jp
from jax.extend import ffi
from mujoco import mjx
from mujoco.mjx.cuda import _engine_core_smooth

ffi.register_ffi_target(
    "kinematics_cuda", _engine_core_smooth.kinematics(), platform="CUDA"
)


def kinematics(m: mjx.Model, d: mjx.Data) -> mjx.Data:
  """Forward kinematics."""
  nbody = m.body_jntadr.shape[-1]
  njnt = m.jnt_type.shape[-1]
  ngeom = m.geom_pos.shape[-2]
  nsite = m.site_pos.shape[-2]
  n = d.qpos.shape[:-1]

  out_types = (
      jax.ShapeDtypeStruct(n + (njnt, 3), dtype=jp.float32),  # xanchor
      jax.ShapeDtypeStruct(n + (njnt, 3), dtype=jp.float32),  # xaxis
      jax.ShapeDtypeStruct(n + (nbody, 3, 3), dtype=jp.float32),  # xmat
      jax.ShapeDtypeStruct(n + (nbody, 3), dtype=jp.float32),  # xpos
      jax.ShapeDtypeStruct(n + (nbody, 4), dtype=jp.float32),  # xquat
      jax.ShapeDtypeStruct(n + (nbody, 3), dtype=jp.float32),  # xipos
      jax.ShapeDtypeStruct(n + (nbody, 3, 3), dtype=jp.float32),  # ximat
      jax.ShapeDtypeStruct(n + (ngeom, 3), dtype=jp.float32),  # geom_xpos
      jax.ShapeDtypeStruct(n + (ngeom, 3, 3), dtype=jp.float32),  # geom_xmat
      jax.ShapeDtypeStruct(n + (nsite, 3), dtype=jp.float32),  # site_xpos
      jax.ShapeDtypeStruct(n + (nsite, 3, 3), dtype=jp.float32),  # site_xmat
  )
  (
      xanchor,
      xaxis,
      xmat,
      xpos,
      xquat,
      xipos,
      ximat,
      geom_xpos,
      geom_xmat,
      site_xpos,
      site_xmat,
  ) = ffi.ffi_call(
      "kinematics_cuda",
      out_types,
      vmap_method="expand_dims",
  )(
      m.qpos0,
      m.body_jntadr,
      m.body_jntnum,
      m.body_parentid,
      m.body_mocapid,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_axis,
      m.jnt_pos,
      m.geom_pos,
      m.geom_quat,
      m.site_pos,
      m.site_quat,
      d.qpos,
      d.mocap_pos,
      d.mocap_quat,
  )
  return d.replace(
      xanchor=xanchor,
      xaxis=xaxis,
      xmat=xmat,
      xpos=xpos,
      xquat=xquat,
      xipos=xipos,
      ximat=ximat,
      geom_xpos=geom_xpos,
      geom_xmat=geom_xmat,
      site_xpos=site_xpos,
      site_xmat=site_xmat,
  )