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
"""Core smooth taichi functions."""

import taichi as ti
import taichi.math as tm
from . import math
from . import types


@ti.kernel
def kinematics(m: ti.template(), d: ti.template()):
  for wid in range(d.nworld):
    # root body
    d.xpos[wid, 0] = tm.vec3(0.0)
    d.xquat[wid, 0] = tm.vec4(1.0, 0.0, 0.0, 0.0)
    d.xipos[wid, 0] = tm.vec3(0.0)
    d.xmat[wid, 0] = ti.Matrix.identity(ti.f32, 3)
    d.ximat[wid, 0] = ti.Matrix.identity(ti.f32, 3)

    for level in ti.static(range(m.nlevel)):  # loop unroll
      # parallel loop over bodies at a given tree level
      for levelid in range(m.level_beg[level], m.level_end[level]):
        bodyid = m.body_bfs[levelid]
        jntadr = m.body_jntadr[bodyid]
        jntnum = m.body_jntnum[bodyid]

        xpos = tm.vec3(0.0, 0.0, 0.0)
        xquat = tm.vec4(1.0, 0.0, 0.0, 0.0)
        if jntnum == 1 and m.jnt_type[jntadr] == 0:
          # free joint
          qadr = m.jnt_qposadr[jntadr]
          xpos = tm.vec3(
            d.qpos[wid, qadr], d.qpos[wid, qadr + 1], d.qpos[wid, qadr + 2])
          xquat = tm.vec4(
            d.qpos[wid, qadr + 3], d.qpos[wid, qadr + 4], d.qpos[wid, qadr + 5],
            d.qpos[wid, qadr + 6])
          d.xanchor[wid, jntadr] = xpos
          d.xaxis[wid, jntadr] = m.jnt_axis[jntadr]
        else:
          # regular or no joints
          # apply fixed translation and rotation relative to parent
          pid = m.body_parentid[bodyid]
          xpos = (d.xmat[wid, pid] @ m.body_pos[bodyid]) + d.xpos[wid, pid]
          xquat = math.mul_quat(d.xquat[wid, pid], m.body_quat[bodyid])

          for _ in range(jntnum):
            qadr = m.jnt_qposadr[jntadr]
            jnt_type = m.jnt_type[jntadr]
            xanchor = math.rot_vec_quat(m.jnt_pos[jntadr], xquat) + xpos
            xaxis = math.rot_vec_quat(m.jnt_axis[jntadr], xquat)

            if jnt_type == 3:  # hinge
              qloc = math.axis_angle_to_quat(
                m.jnt_axis[jntadr], d.qpos[wid, qadr] - m.qpos0[qadr])
              xquat = math.mul_quat(xquat, qloc)
              # correct for off-center rotation
              xpos = xanchor - math.rot_vec_quat(m.jnt_pos[jntadr], xquat)

            d.xanchor[wid, jntadr] = xanchor
            d.xaxis[wid, jntadr] = xaxis
            jntadr += 1

        d.xpos[wid, bodyid] = xpos
        d.xquat[wid, bodyid] = tm.normalize(xquat)
        d.xmat[wid, bodyid] = math.quat_to_mat(xquat)
