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
"""Core smooth warp functions."""

import warp as wp
import jax
from jax import numpy as jp
from mujoco import mjx
import numpy as np
from warp.jax_experimental import jax_kernel
from mujoco.mjx._src import test_util


@wp.func
def quat_to_mat(q: wp.array(dtype=float)):
  """Converts a quaternion into a 9-dimensional rotation matrix."""
  q = wp.outer(q, q)

  return wp.array([
      [
          q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
          2 * (q[1, 2] - q[0, 3]),
          2 * (q[1, 3] + q[0, 2]),
      ],
      [
          2 * (q[1, 2] + q[0, 3]),
          q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
          2 * (q[2, 3] - q[0, 1]),
      ],
      [
          2 * (q[1, 3] - q[0, 2]),
          2 * (q[2, 3] + q[0, 1]),
          q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
      ],
  ])


@wp.kernel
def _kinematics(
  # static (at least they're supposed to be)
  nq: wp.array(dtype=int),
  nbody: wp.array(dtype=int),
  njnt: wp.array(dtype=int),
  ngeom: wp.array(dtype=int),
  nsite: wp.array(dtype=int),
  nmocap: wp.array(dtype=int),
  # inputs
  qpos0: wp.array(dtype=float),
  body_jntadr: wp.array(dtype=int),
  body_jntnum: wp.array(dtype=int),
  body_parentid: wp.array(dtype=int),
  body_mocapid: wp.array(dtype=int),
  body_pos: wp.array2d(dtype=float),
  body_quat: wp.array2d(dtype=float),
  body_ipos: wp.array2d(dtype=float),
  body_iquat: wp.array2d(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_axis: wp.array(dtype=float),
  jnt_pos: wp.array2d(dtype=float),
  geom_pos: wp.array2d(dtype=float),
  geom_quat: wp.array2d(dtype=float),
  site_pos: wp.array2d(dtype=float),
  site_quat: wp.array2d(dtype=float),
  qpos: wp.array(dtype=float),
  mocap_pos: wp.array2d(dtype=float),
  mocap_quat: wp.array2d(dtype=float),
  # outputs
  xanchor: wp.array2d(dtype=float),
  xaxis: wp.array2d(dtype=float),
  xmat: wp.array3d(dtype=float),
  xpos: wp.array2d(dtype=float),
  xquat: wp.array2d(dtype=float),
  xipos: wp.array2d(dtype=float),
  ximat: wp.array3d(dtype=float),
  geom_xpos: wp.array2d(dtype=float),
  geom_xmat: wp.array3d(dtype=float),
  site_xpos: wp.array2d(dtype=float),
  site_xmat: wp.array2d(dtype=float),
):
  # This is a stupid version, transcribed from a naive CUDA version
  # a second iteration of this should parallelize over kinmeatic trees
  # similar to warp/genesis.
  # Note: MJX does a custom vmap over tree levels, mostly as a workaround
  # to static shape requirements.
  tid = wp.tid()

  # qpos = qpos[tid]  # this does not work...no idea why

  mocap_pos = mocap_pos[tid]
  mocap_quat = mocap_quat[tid]
  xanchor = xanchor[tid]
  xaxis = xaxis[tid]
  xmat = xmat[tid]
  xpos = xpos[tid]
  xquat = xquat[tid]
  xipos = xipos[tid]
  ximat = ximat[tid]
  geom_xpos = geom_xpos[tid]
  geom_xmat = geom_xmat[tid]
  site_xpos = site_xpos[tid]
  site_xmat = site_xmat[tid]

  # set world position and orientation
  xpos[0] = 0.0; xpos[1] = 0.0; xpos[2] = 0.0
  xquat[0] = 0.0; xquat[1] = 0.0; xquat[2] = 0.0; xquat[3] = 0.0
  # TODO: there has got to be a cleaner way to do this, please help
  for i in range(3):
    xipos[i] = 0.0
  for i in range(3):
    for j in range(3):
      xmat[i, j] = 0.0
      ximat[i, j] = 0.0
  for i in range(3):
    xmat[i, i] = 1.0
    ximat[i, i] = 1.0
  xquat[0] = 1.0

  lxpos = wp.vec3(0.0, 0.0, 0.0)
  lxquat = wp.vec4(0.0, 0.0, 0.0, 0.0)
  for i in range(1, nbody[0]):
    # TODO: access static array...is there a better way to pass these in?
    jntadr = body_jntadr[i]  
    jntnum = body_jntnum[i]

    # free joint
    if (
      (jntnum == 1) and
      # TODO: need int() since warp parsing breaks
      (jnt_type[jntadr] == int(mjx.JointType.FREE.value))
    ):
      # get qpos address
      qadr = int(jnt_qposadr[jntadr])

      # copy pos and quat from qpos
      for j in range(3):
        lxpos[j] = qpos[tid][int(qadr + j)]  # TODO: please make this cleaner
      for j in range(4):
        lxquat[j] = qpos[tid][int(qadr + 3 + j)]

      wp.normalize(lxquat)

      # assign xanchor and xaxis
      for j in range(3):
        xanchor[jntadr, j] = lxpos[j]
        xaxis[jntadr, j] = jnt_axis[jntadr, j]
    # else:
    #   pid = body_parentid[i]
    #   # get body pos and quat: from model or mocap
    #   bodypos = wp.vec3(0.0, 0.0, 0.0)
    #   bodyquat = wp.vec4(1.0, 0.0, 0.0, 0.0)
    #   quat = wp.vec4(1.0, 0.0, 0.0, 0.0)
    #   if body_mocapid[i] >= 0:
    #     bodypos = mocap_pos[body_mocapid[i]]
    #     bodyquat = mocap_quat[body_mocapid[i]]
    #   else:
    #     bodypos = body_pos[i]
    #     bodyquat = body_quat[i]

    #   // apply fixed translation and rotation relative to parent
    #   if (pid) {
    #     mulMatVec3(lxpos, xmat + 9 * pid, bodypos);
    #     addTo(lxpos, xpos + 3 * pid, 3);
    #     mulQuat(lxquat, xquat + 4 * pid, bodyquat);
    #   } else {
    #     // parent is the world
    #     copy(lxpos, bodypos, 3);
    #     copy(lxquat, bodyquat, 4);
    #   }

    #   // accumulate joints, compute xpos and xquat for this body
    #   float lxanchor[3], lxaxis[3];
    #   for (int j = 0; j < jntnum; j++) {
    #     // get joint id, qpos address, joint type
    #     int jid = jntadr + j;
    #     int qadr = jnt_qposadr[jid];
    #     int jtype = jnt_type[jid];

    #     // compute axis in global frame; ball jnt_axis is (0,0,1), set by
    #     // compiler
    #     rotVecQuat(lxaxis, jnt_axis + 3 * jid, lxquat);

    #     // compute anchor in global frame
    #     rotVecQuat(lxanchor, jnt_pos + 3 * jid, lxquat);
    #     addTo(lxanchor, lxpos, 3);

    #     // apply joint transformation
    #     switch (jtype) {
    #       case mjJNT_SLIDE:
    #         addToScl(lxpos, lxaxis, qpos[qadr] - qpos0[qadr], 3);
    #         break;

    #       case mjJNT_BALL:
    #       case mjJNT_HINGE: {
    #         // compute local quaternion rotation
    #         float qloc[4];
    #         if (jtype == mjJNT_BALL) {
    #           copy(qloc, qpos + qadr, 4);
    #           normalize(qloc, 4);
    #         } else {
    #           axisAngle2Quat(qloc, jnt_axis + 3 * jid,
    #                          qpos[qadr] - qpos0[qadr]);
    #         }

    #         // apply rotation
    #         mulQuat(lxquat, lxquat, qloc);

    #         // correct for off-center rotation
    #         float vec[3];
    #         rotVecQuat(vec, jnt_pos + 3 * jid, lxquat);
    #         sub(lxpos, lxanchor, vec, 3);
    #       } break;

    #       default:
    #         // TODO: whatever cuda error semantics are
    #         // mjERROR("unknown joint type %d", jtype);  // SHOULD NOT OCCUR
    #         break;
    #     }

    #     // assign xanchor and xaxis
    #     copy(xanchor + 3 * jid, lxanchor, 3);
    #     copy(xaxis + 3 * jid, lxaxis, 3);
    #   }

    # assign xquat and xpos, construct xmat
    xquat[i] = wp.normalize(xquat[i])
    xpos[i] = lxpos
    xmat[i] = quat_to_mat(lxquat)


def kinematics(m: mjx.Model, d: mjx.Data) -> mjx.Data:
  """Forward kinematics."""
  # we must convert static args to wp.array since `jax_kernel`
  #  does not support static args
  nq = jp.array([m.nq], dtype=int)
  nbody = jp.array([m.nbody], dtype=int)
  njnt = jp.array([m.njnt], dtype=int)
  ngeom = jp.array([m.ngeom], dtype=int)
  nsite = jp.array([m.nsite], dtype=int)
  nmocap = jp.array([m.nmocap], dtype=int)

  jax_kinematics = jax_kernel(_kinematics)
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
    site_xmat) =  jax_kinematics(
      nq, nbody, njnt, ngeom, nsite, nmocap,
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


batch_size = 2
m = test_util.load_test_file('humanoid/10_humanoids.xml')
@jax.vmap
def make_model_and_data(val):
    mx = mjx.put_model(m)
    dx = mjx.make_data(m)
    return mx, dx

# naively add a batch dim...but this isn't realistic
mx, dx = make_model_and_data(jp.arange(batch_size))
mx = mjx.put_model(m)
dx = mjx.make_data(m)
# the JIT would be within some outer function, `jax_kernel` would
# need to support batch tracer objects
jax.jit(kinematics)(mx, dx)












# if __name__ == '__main__':
#   m = test_util.load_test_file('humanoid/10_humanoids.xml')
#   mx = mjx.put_model(m)
#   dx = mjx.make_data(m)

#   dx_x = jax.jit(mjx.kinematics)(mx, dx)
#   m, d = mx, dx









# data = list(range(10))
# @wp.kernel
# def scale(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
#     i = wp.tid()
#     a = wp.zeros(3)
#     if s[0] == 0:
#       out[i] = x[i] + a[0]
#       return
#     out[i] = x[i] * 2.0
# print(jax.jit(jax_kernel(scale))(jp.array(data, dtype=jp.float32), jp.array([0])))

# N = 10
# a_np = np.random.rand(N).astype(np.float32)
# a_wp = wp.array(a_np, dtype=wp.float32)
# c_np = np.zeros(N, dtype=np.float32)
# c_wp = wp.array(c_np, dtype=wp.float32)
# wp.launch(
#     kernel=scale, dim=N, inputs=[a_wp, wp.array([2], dtype=int)],
#     outputs=[c_wp], device="cuda"
# )

# result = wp.from_numpy(c_wp, dtype=np.float32).numpy()
# print(result)









# @wp.kernel
# def simple_kernel(a: wp.array2d(dtype=wp.float32),
#                   b: wp.array2d(dtype=wp.float32),
#                   c: wp.array1d(dtype=wp.float32)):
#     tid = wp.tid()
#     r = 0.0
#     tmp = wp.vector(length=3, dtype=float)
#     tmp[0] = a[tid][0]
#     tmp[1] = a[tid][1]
#     tmp[2] = a[tid][2]
#     b = b[tid]

#     for i in range(3):
#         r += tmp[i] * b[i]
#     c[tid] = r
# N = 1024
# a_np = np.random.rand(N, 3).astype(np.float32)
# b_np = np.random.rand(N, 3).astype(np.float32)
# c_np = np.zeros(N, dtype=np.float32)
# a_wp = wp.array(a_np, dtype=wp.float32)
# b_wp = wp.array(b_np, dtype=wp.float32)
# c_wp = wp.array(c_np, dtype=wp.float32)
# wp.launch(
#     kernel=simple_kernel, dim=N, inputs=[a_wp, b_wp],
#     outputs=[c_wp], device="cuda"
# )

# result = wp.from_numpy(c_wp, dtype=np.float32).numpy()
# print(result)


# # @wp.kernel
# # def scale(x: wp.array(dtype=Any), s: Any):
# #     i = wp.tid()
# #     x[i] = s * x[i]

# # data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# # n = len(data)

# # x32 = wp.array(data, dtype=wp.float32)

# # wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])

# # jax.jit(jax_kernel(scale))(jp.array(data), n)

# # @wp.kernel
# # def scale(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32)):
# #     i = wp.tid()
# #     x[i] = s[0] * x[i]
# # jax.jit(jax_kernel(scale))(jp.array(data), np.array([n]))

# # @wp.kernel
# # def scale(x: wp.array(dtype=wp.float32), s: wp.float32):
# #     i = wp.tid()
# #     x[i] = s * x[i]
# # jax.jit(jax_kernel(scale))(jp.array(data), n)


# data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# n = len(data)

# @wp.kernel
# def scale(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32), static: int, out: wp.array(dtype=wp.float32)):
#     i = wp.tid()
#     out[i] = s[0] * x[i] + float(static)
# print(jax.jit(jax_kernel(scale))(jp.array(data, dtype=jp.float32), jp.array([n], dtype=jp.float32), 2))


# @wp.kernel
# def scale(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
#     i = wp.tid()
#     out[i] = s[0] * x[i]


# @wp.kernel
# def scale(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32), static: list[int], out: wp.array(dtype=wp.float32)):
#     i = wp.tid()
#     out[i] = s[0] * x[i] + float(static)

# N = 1024
# a_np = np.random.rand(N).astype(np.float32)
# a_wp = wp.array(a_np, dtype=wp.float32)
# c_np = np.zeros(N, dtype=np.float32)
# b_wp = wp.array([1], dtype=wp.float32)
# c_wp = wp.array(c_np, dtype=wp.float32)
# wp.launch(
#     kernel=scale, dim=N, inputs=[a_wp, b_wp, wp.constant(2)],
#     outputs=[c_wp], device="cuda"
# )

# result = wp.from_numpy(c_wp, dtype=np.float32).numpy()
# print(result)





# data = list(range(10))
# assert isinstance(mjx.JointType.FREE.value, int)

# @wp.kernel
# def scale(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
#     i = wp.tid()
#     print(mjx.JointType.FREE.value)
#     if s[0] == mjx.JointType.FREE.value:  # works if I compare to 0 directly
#       out[i] = x[i]
#       return
#     out[i] = x[i] * 2.0

# print(jax.jit(jax_kernel(scale))(jp.array(data, dtype=jp.float32), jp.array([0])))
