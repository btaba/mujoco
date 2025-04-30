"""Minimal example."""

import os
import jax
import warp as wp
from warp.jax_experimental import ffi as warp_ffi
import mujoco_warp as mjwarp
import mujoco_warp._src.warp_util as warp_util
import jax
import mujoco
import warp as wp
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import test_util

from mujoco.mjx._src import types


@warp_util.kernel
def _root(
  qpos: wp.array2d(dtype=float),
  xpos: wp.array2d(dtype=wp.vec3),
):
  worldid = wp.tid()
  xpos[worldid, 0] = wp.vec3(qpos[worldid, 0])


def kinematics_(
  qpos: wp.array2d(dtype=float),
  xpos: wp.array2d(dtype=wp.vec3),
):
  """Forward kinematics."""

  nworld = wp.static(qpos.shape[0])
  wp.launch(_root, dim=(nworld), inputs=[qpos], outputs=[xpos])


def _kinematics_shim(
    qpos: wp.array1d(dtype=float),
    xpos: wp.array1d(dtype=wp.vec3),
):
  args = (
      qpos,
      xpos,
  )
  new_args = [None] * len(args)
  annotations = tuple(kinematics_.__annotations__.items())
  for i in range(len(args)):
    expected_ndim = annotations[i][1].ndim

    # remove the expanded dims, assuming we used the `expand_dims` vmap_method
    if expected_ndim < args[i].ndim:
      naxes = args[i].ndim - expected_ndim
      # check that we are squeezing size 1 axes only
      assert args[i].shape[:naxes] == (1,) * naxes
      new_args[i] = args[i].reshape(args[i].shape[naxes:])
      new_args[i].ndim = args[i].ndim - naxes
      continue

    # add a batch dim for fields that were not vmapped
    if expected_ndim > args[i].ndim:
      extra_dims = expected_ndim - args[i].ndim
      new_args[i] = args[i].reshape((1,) * extra_dims  + args[i].shape)
      new_args[i].ndim = expected_ndim
      continue

    new_args[i] = args[i]
  kinematics_(*new_args)




def kinematics(m: types.Model, d: types.Data) -> types.Data:
  """Forward kinematics."""
  output_dims = {
      'xpos': (m.nbody, 3),
  }
  jf = warp_ffi.jax_callable(
      _kinematics_shim, num_outputs=1,
      output_dims=output_dims,
      vmap_method='expand_dims',
      graph_compatible=True,
  )
  return jf(d.qpos)


if __name__ == '__main__':
  wp.clear_kernel_cache()

  # wp.config.verbose = True
  # wp.config.print_launches = True
  # wp.config.mode = 'debug'
  # wp.config.verify_cuda = True

  os.environ['MJX_WARP_ENABLED'] = 'true'
  m = test_util.load_test_file('pendula.xml')
  d = mujoco.MjData(m)
  mx = mjx.put_model(m, backend_impl='warp')

  rng = jax.random.PRNGKey(0)
  dx = mjx.make_data(m, backend_impl='warp')
  rng, key = jax.random.split(rng)
  qpos = jax.random.uniform(key, (m.nq,))
  rng, key1, key2 = jax.random.split(rng, 3)
  mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
  mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
  mocap_quat = math.normalize(mocap_quat)
  dx = dx.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

  # dx = kinematics(mx, dx)
  dx = jax.jit(kinematics)(mx, dx)  # segfault!
