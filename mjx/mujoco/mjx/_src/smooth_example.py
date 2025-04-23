"""Core smooth dynamics functions, example  calling into Warp."""
import jax
import jax.numpy as jp
import mujoco

# pip install -e . # in the top-level mjx directory
from mujoco import mjx
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model

# pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
# pip install mujoco_warp
import warp as wp
from mujoco_warp._src import math
from mujoco_warp._src.warp_util import kernel
from warp.jax_experimental.ffi import jax_callable


"""
Warp code, copied below for convenience
"""
@kernel
def _root(
  xpos: wp.array2d(dtype=wp.vec3),
  xquat: wp.array2d(dtype=wp.quat),
  xmat: wp.array2d(dtype=wp.mat33),
  xipos: wp.array2d(dtype=wp.vec3),
  ximat: wp.array2d(dtype=wp.mat33),
):
  worldid = wp.tid()
  xpos[worldid, 0] = wp.vec3(0.0)
  xquat[worldid, 0] = wp.quat(1.0, 0.0, 0.0, 0.0)
  xipos[worldid, 0] = wp.vec3(0.0)
  xmat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)
  ximat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)

@kernel
def _level(
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
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_pos: wp.array(dtype=wp.vec3),
  jnt_axis: wp.array(dtype=wp.vec3),
  # Data Inputs
  qpos_in: wp.array2d(dtype=float),
  xpos_in: wp.array2d(dtype=wp.vec3),
  xquat_in: wp.array2d(dtype=wp.quat),
  xmat_in: wp.array2d(dtype=wp.mat33),
  # Static Kernel Args
  leveladr: int,
  # Data Outputs
  xpos_out: wp.array2d(dtype=wp.vec3),
  xquat_out: wp.array2d(dtype=wp.quat),
  xmat_out: wp.array2d(dtype=wp.mat33),
  xipos_out: wp.array2d(dtype=wp.vec3),
  ximat_out: wp.array2d(dtype=wp.mat33),
  xanchor_out: wp.array2d(dtype=wp.vec3),
  xaxis_out: wp.array2d(dtype=wp.vec3),
):
  worldid, nodeid = wp.tid()
  bodyid = body_tree[leveladr + nodeid]
  jntadr = body_jntadr[bodyid]
  jntnum = body_jntnum[bodyid]
  qpos = qpos_in[worldid]

  xpos = wp.vec3(0.0)
  xquat = wp.quat(1.0, 0.0, 0.0, 0.0)

  if jntnum == 0:
    # no joints - apply fixed translation and rotation relative to parent
    pid = body_parentid[bodyid]
    xpos = (xmat_in[worldid, pid] * body_pos[bodyid]) + xpos_in[worldid, pid]
    xquat = math.mul_quat(xquat_in[worldid, pid], body_quat[bodyid])
  elif jntnum == 1 and jnt_type[jntadr] == wp.static(JointType.FREE.value):
    # free joint
    qadr = jnt_qposadr[jntadr]
    xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
    xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
    xanchor_out[worldid, jntadr] = xpos
    xaxis_out[worldid, jntadr] = jnt_axis[jntadr]
  else:
    # regular joints
    pid = body_parentid[bodyid]
    xpos = (xmat_in[worldid, pid] * body_pos[bodyid]) + xpos_in[worldid, pid]
    xquat = math.mul_quat(xquat_in[worldid, pid], body_quat[bodyid])

    for _ in range(jntnum):
      qadr = jnt_qposadr[jntadr]
      jnt_type_val = jnt_type[jntadr]
      jnt_axis_val = jnt_axis[jntadr]
      xanchor = math.rot_vec_quat(jnt_pos[jntadr], xquat) + xpos
      xaxis = math.rot_vec_quat(jnt_axis_val, xquat)

      if jnt_type_val == wp.static(JointType.BALL.value):
        qloc = wp.quat(
          qpos[qadr + 0],
          qpos[qadr + 1],
          qpos[qadr + 2],
          qpos[qadr + 3],
        )
        xquat = math.mul_quat(xquat, qloc)
        # correct for off-center rotation
        xpos = xanchor - math.rot_vec_quat(jnt_pos[jntadr], xquat)
      elif jnt_type_val == wp.static(JointType.SLIDE.value):
        xpos += xaxis * (qpos[qadr] - qpos0[qadr])
      elif jnt_type_val == wp.static(JointType.HINGE.value):
        qpos0_val = qpos0[qadr]
        qloc = math.axis_angle_to_quat(jnt_axis_val, qpos[qadr] - qpos0_val)
        xquat = math.mul_quat(xquat, qloc)
        # correct for off-center rotation
        xpos = xanchor - math.rot_vec_quat(jnt_pos[jntadr], xquat)

      xanchor_out[worldid, jntadr] = xanchor
      xaxis_out[worldid, jntadr] = xaxis
      jntadr += 1

  xpos_out[worldid, bodyid] = xpos
  xquat = wp.normalize(xquat)
  xquat_out[worldid, bodyid] = xquat
  xmat_out[worldid, bodyid] = math.quat_to_mat(xquat)
  xipos_out[worldid, bodyid] = xpos + math.rot_vec_quat(body_ipos[bodyid], xquat)
  ximat_out[worldid, bodyid] = math.quat_to_mat(
    math.mul_quat(xquat, body_iquat[bodyid])
  )


@kernel
def geom_local_to_global(
  # Model Inputs
  geom_bodyid: wp.array(dtype=int),
  geom_pos: wp.array(dtype=wp.vec3),
  geom_quat: wp.array(dtype=wp.quat),
  # Data Inputs
  xpos: wp.array2d(dtype=wp.vec3),
  xquat: wp.array2d(dtype=wp.quat),
  # Output Args
  geom_xpos_out: wp.array2d(dtype=wp.vec3),
  geom_xmat_out: wp.array2d(dtype=wp.mat33),
):
  worldid, geomid = wp.tid()
  bodyid = geom_bodyid[geomid]
  xpos_b = xpos[worldid, bodyid]
  xquat_b = xquat[worldid, bodyid]
  geom_xpos_out[worldid, geomid] = xpos_b + math.rot_vec_quat(geom_pos[geomid], xquat_b)
  geom_xmat_out[worldid, geomid] = math.quat_to_mat(
    math.mul_quat(xquat_b, geom_quat[geomid])
  )


def kinematics_(
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
  # TODO(btaba): test batching on `geom_pos` as an example as well
  geom_pos: wp.array(dtype=wp.vec3),
  geom_quat: wp.array(dtype=wp.quat),
  site_bodyid: wp.array(dtype=int),
  # Data Inputs
  qpos: wp.array2d(dtype=float),
  # Outputs
  xpos: wp.array2d(dtype=wp.vec3),
  xquat: wp.array2d(dtype=wp.quat),
  xmat: wp.array2d(dtype=wp.mat33),
  xipos: wp.array2d(dtype=wp.vec3),
  ximat: wp.array2d(dtype=wp.mat33),
  xanchor: wp.array2d(dtype=wp.vec3),
  xaxis: wp.array2d(dtype=wp.vec3),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
):
  nworld = qpos.shape[0]
  nbody = body_parentid.shape[0]
  ngeom = geom_bodyid.shape[0]

  # Launch kernels with explicit arguments
  wp.launch(_root, dim=(nworld), inputs=[], outputs=[xpos, xquat, xmat, xipos, ximat])

  body_treeadr_np = body_treeadr.numpy()
  for i in range(1, len(body_treeadr_np)):
    beg = body_treeadr_np[i]
    end = nbody if i == len(body_treeadr_np) - 1 else body_treeadr_np[i + 1]
    wp.launch(
        _level,
        dim=(nworld, end - beg),
        inputs=[
          body_tree, qpos0, body_parentid, body_jntadr, body_jntnum,
          body_pos, body_quat, body_ipos, body_iquat, jnt_type,
          jnt_qposadr, jnt_pos, jnt_axis, qpos,
          xpos, xquat, xmat,
          beg # Static arg
        ],
        outputs=[xpos, xquat, xmat, xipos, ximat, xanchor, xaxis]
    )

  if ngeom > 0:
    wp.launch(
        geom_local_to_global,
        dim=(nworld, ngeom),
        inputs=[geom_bodyid, geom_pos, geom_quat, xpos, xquat],
        outputs=[geom_xpos, geom_xmat]
    )



"""
Code in MJX that will call into Warp code.
"""

def _kinematics_warp(m: Model, d: Data) -> Data:
  """This would live in MJX."""
  mjwarp_kinematics_jax = jax_callable(
    kinematics_,
    num_outputs=9,
    vmap_method='broadcast_all',
    output_dims={
      'xpos': (m.nbody, 3),
      'xquat': (m.nbody, 4),
      'xmat': (m.nbody, 9),
      'xipos': (m.nbody, 3),
      'ximat': (m.nbody, 9),
      'xanchor': (m.njnt, 3),
      'xaxis': (m.njnt, 3),
      'geom_xpos': (m.ngeom, 3),
      'geom_xmat': (m.ngeom, 9),
    }
  )
  out = mjwarp_kinematics_jax(
    m._blob.body_tree.numpy(),
    m.qpos0,
    m.body_parentid,
    m.body_jntadr,
    m.body_jntnum,
    m.body_pos,
    m.body_quat,
    m.body_ipos,
    m.body_iquat,
    m._blob.body_treeadr.numpy(),
    m.jnt_type,
    m.jnt_qposadr,
    m.jnt_pos,
    m.jnt_axis,
    m.geom_bodyid,
    m.geom_pos,
    m.geom_quat,
    m.site_bodyid,
    # Data Inputs
    jp.expand_dims(d.qpos, 0),  # TODO(btaba): expand_dims if not called in a vmap
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
  )
  return d


def kinematics(m: Model, d: Data) -> Data:
  """Top-level function exposed in MJX."""
  return _kinematics_warp(m, d)


"""
Examples
"""
BATCH_SIZE = 8

m = mujoco.MjModel.from_xml_path('./mujoco/mjx/test_data/humanoid/humanoid.xml')
mx = mjx.put_model(m, backend_impl='warp')
dx = mjx.make_data(mx, backend_impl='warp')
rng = jax.random.PRNGKey(0)
keys = jax.random.split(rng, BATCH_SIZE)
dx_batch = jax.vmap(lambda rng: dx.replace(qpos=dx.qpos * jax.random.uniform(rng, (1,))))(keys)


jax.jit(jax.vmap(kinematics, in_axes=(None, 0)))(mx, dx_batch)

# we also need a way to expand_dims if not called in vmap
jax.jit(kinematics)(mx, dx)



import jax
import jax.numpy as jp
import warp as wp
from warp.jax_experimental.ffi import jax_callable

@wp.kernel
def scale_kernel(a: wp.array2d(dtype=float), s: float, output: wp.array2d(dtype=float)):
    wid, tid = wp.tid()
    output[wid, tid] = a[wid, tid] * s

def example_func(
    a: wp.array2d(dtype=float),
    s: float,
    c: wp.array2d(dtype=float),
):
  wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])

def jax_func(a: jax.Array, s: float):
  jf = jax_callable(example_func, num_outputs=1, vmap_method="broadcast_all")
  return jf(a, s)

# c = jax.jit(jax_func, static_argnums=(1,))(jp.ones((10, 10)), 2)  # works!
c = jax.jit(jax.vmap(jax_func, in_axes=(0, None)), static_argnums=(1,))(jp.ones((10, 10, 10)), 2)  # does not work