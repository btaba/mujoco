"""Minimal example."""

import os
import jax
import jax.numpy as jp
import jax
import numpy as np
import mujoco
import warp as wp
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import test_util
from mujoco.mjx._src import types

import mujoco_warp as mjwarp
import mujoco_warp._src.warp_util as warp_util

import warp as wp
from warp.jax_experimental import ffi as warp_ffi


@warp_util.kernel
def _kinematics_kernel(
  # Model:
  mocap_bodyid: wp.array(dtype=int),
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xquat_in: wp.array2d(dtype=wp.quat),
  efc_active_in: wp.array(dtype=bool),
  # Data out:
  nefc_out: wp.array(dtype=int),
  xpos_out: wp.array2d(dtype=wp.vec3),
):
  worldid = wp.tid()
  nefc_out[worldid] = 0
  wp.printf("worldid %d\n", worldid)
  for i in range(xpos_in.shape[1]):
    tmp = xquat_in[worldid, i]
    if efc_active_in[worldid]:
      wp.atomic_add(nefc_out, worldid, 1)
    xpos_out[worldid, i] = xpos_in[worldid, i] + wp.vec3(tmp.x, tmp.y, tmp.z)

  for i in range(mocap_bodyid.shape[0]):
    wp.printf("worldid %d, mocap_bodyid %d\n", worldid, mocap_bodyid[i])
    xpos_out[worldid, mocap_bodyid[i]] = wp.vec3(-1.0)


def kinematics_warp(
  mocap_bodyid: wp.array(dtype=int),
  xpos_in: wp.array2d(dtype=wp.vec3),
  xquat_in: wp.array2d(dtype=wp.quat),
  efc_active_in: wp.array(dtype=bool),
  nefc_out: wp.array(dtype=int),
  xpos_out: wp.array2d(dtype=wp.vec3),
):
  """Forward kinematics."""

  nworld = xpos_in.shape[0]
  wp.launch(
    _kinematics_kernel, dim=(nworld),
    inputs=[mocap_bodyid, xpos_in, xquat_in, efc_active_in],
    outputs=[nefc_out, xpos_out]
  )


def _kinematics_shim(
    mocap_bodyid: wp.array(dtype=int),
    xpos_in: wp.array(dtype=wp.vec3),
    xquat_in: wp.array(dtype=wp.quat),
    efc_active_in: wp.array(dtype=bool),
    nefc_out: wp.array(dtype=int),
    xpos_out: wp.array(dtype=wp.vec3),
):
  args = (
      mocap_bodyid,
      xpos_in,
      xquat_in,
      efc_active_in,
      nefc_out,
      xpos_out,
  )
  print("\n--- Inside _kinematics_shim (traced by JAX) ---")
  print(f"  efc_active_in (JAX DeviceArray): Type={type(efc_active_in)}, Shape={efc_active_in.shape}, Device={efc_active_in.device}")
  device_address = efc_active_in.ptr
  print(f"  --> Device memory address of efc_active_in: {hex(device_address)}")

  names = ['mocap_bodyid', 'xpos_in', 'xquat', 'efc__active', 'nefc', 'xpos_out']
  new_args = [None] * len(args)
  annotations = tuple(_kinematics_kernel.__annotations__.items())
  for i in range(len(args)):
    expected_ndim = annotations[i][1].ndim
    print('Expected ndim for ', names[i], ': ', expected_ndim)

    # Remove the expanded_dim if we are exceeding the expected ndim.
    # i.e. Unbatched model fields will get an extra dim due to
    # vmap_method="expand_dims".
    if args[i].ndim > expected_ndim and args[i].shape[0] == 1:
      extra_dim = args[i].ndim - expected_ndim
      assert sum(args[i].shape[:extra_dim]) == extra_dim
      arg = args[i].reshape(args[i].shape[extra_dim:])
      arg.ndim  = expected_ndim
      new_args[i] = arg
      print('Removing extra dim: ', names[i], args[i].shape, '=>', new_args[i].shape)
      continue

    # Squash nested vmap batch axes.
    if args[i].ndim > expected_ndim:
      extra_ndim = args[i].ndim - expected_ndim
      new_args[i] = args[i].reshape((-1,) + args[i].shape[extra_ndim + 1:])
      new_args[i].ndim = expected_ndim
      print('Squashing extra dim: ', names[i], args[i].shape, '=>', new_args[i].shape)
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
      print('Leading batch dim of 1, adding stride: ', names[i], old_strides, '=>', new_args[i].strides)
      continue

    # Add batch dims if they don't exist. This occurs when the underlying
    # function expects a batch dim but the outer function was not called
    # with vmap.
    # e.g. Model/Data have nworld == 1, but without the leading dim in JAX.
    if expected_ndim > args[i].ndim:
      extra_dims = expected_ndim - args[i].ndim
      new_args[i] = args[i].reshape((1,) * extra_dims  + args[i].shape)
      new_args[i].ndim = expected_ndim
      print('No leading batch dims', names[i], args[i].shape, '=>', new_args[i].shape)
      continue

    new_args[i] = args[i]
    print('Did nothing: ', names[i], args[i].shape, '=>', new_args[i].shape)

  print('NEW ARGS', {k: new_args[i] for i, k in enumerate(names)})
  kinematics_warp(*new_args)


def kinematics_jax(m: types.Model, d: types.Data):
  """Forward kinematics."""
  output_dims = {
      'nefc_out': (1,),
      'xpos_out': (21, 3),
  }
  jf = warp_ffi.jax_callable(
      _kinematics_shim, num_outputs=2,
      output_dims=output_dims,
      vmap_method='expand_dims',
      graph_compatible=True,
  )
  return jf(m._impl.mocap_bodyid,
            d.xpos, d.xquat, d._impl.efc__active)


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
  dx = dx.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat,
                  xquat=dx.xquat + mocap_quat[0],
                  xpos=dx.xpos + mocap_pos[0])
  dx = dx.tree_replace({'_impl.efc__active': np.random.binomial(1, 0.5, 120) == 1})

  # Vanilla call.
  out = jax.jit(kinematics_jax)(mx, dx)

  # Vmapped on data.
  batch_size = 8
  def make_data(rng):
    rng, key = jax.random.split(rng)
    qpos = jax.random.uniform(key, (m.nq,))
    rng, key1, key2 = jax.random.split(rng, 3)
    mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
    mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
    mocap_quat = math.normalize(mocap_quat)
    return dx.replace(qpos=qpos, mocap_pos=mocap_pos,
                      mocap_quat=mocap_quat, xquat=dx.xquat + mocap_quat[0])

  rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
  dx_batch = jax.vmap(make_data)(rng)
  out = jax.jit(jax.vmap(kinematics_jax, in_axes=[None, 0]))(mx, dx_batch)

  # Vmapped on data with in_axes.
  in_axes = jax.tree_map(lambda x: 0, dx)
  in_axes = in_axes.replace(xquat=None)
  dx_batch = dx_batch.replace(xquat=-100 * dx.xquat)
  out = jax.jit(jax.vmap(kinematics_jax, in_axes=[None, in_axes]))(mx, dx_batch)

  # Nested vmap.
  dx_batch = jax.vmap(make_data)(rng)
  dx_batch2 = jax.tree_map(lambda x: x.reshape((2, 4) + x.shape[1:]), dx_batch)
  out = jax.jit(jax.vmap(jax.vmap(kinematics_jax, in_axes=[None, 0]), in_axes=[None, 0]))(mx, dx_batch2)
