"""Custom vmap example."""

import jax
import flax
from jax import numpy as jp
from jax._src.interpreters.batching import register_vmappable
import numpy as np
import warp as wp
from warp.jax_experimental import ffi
from mujoco.mjx._src import test_util


@flax.struct.dataclass
class Model:
  nq: int


@flax.struct.dataclass
class Data:
  qpos: jax.Array  # (nworld, nq)
  ncon: jax.Array  # (1,)
  nefc: jax.Array  # (1,)
  shape = property(lambda self: self.qpos.shape)


# TODO(btaba): make this more codegen friendly.
def to_elt(cont, _, d, axis):
  return Data(cont(d.qpos, axis), d.ncon, d.nefc)
def from_elt(cont, axis_size, d, axis_dest):
  return Data(cont(axis_size, d.qpos, axis_dest), d.ncon, d.nefc)
register_vmappable(Data, int, int, to_elt, from_elt, None)


@wp.kernel
def _kernel(
  qpos_in: wp.array2d(dtype=float),
  ncon: wp.array(dtype=int),
  nefc: wp.array(dtype=int),
  qpos_out: wp.array2d(dtype=float),
):
  worldid = wp.tid()
  ncon[0] = ncon[0] + 1
  nefc[0] = nefc[0] - 1
  qpos_out[worldid, 0] = qpos_in[worldid, 0] * wp.float(ncon[0])


def _wp_launch(
  qpos_in: wp.array2d(dtype=float),
  ncon: wp.array(dtype=int),
  nefc: wp.array(dtype=int),
  qpos_out: wp.array2d(dtype=float),
):
  print('nefc: ', nefc.ptr)
  print('ncon: ', ncon.ptr)
  wp.launch(_kernel, dim=(qpos_in.shape[0],), inputs=[qpos_in], outputs=[ncon, nefc, qpos_out])


def _shim(
  qpos: wp.array(dtype=float),
  ncon: wp.array(dtype=int),
  nefc: wp.array(dtype=int),
):
  if ncon.ndim > 1:
    ncon.ndim -= 1  # remove expanded dims
    ncon = ncon.reshape((-1,) + ncon.shape[2:])
  if nefc.ndim > 1:
    nefc.ndim -= 1  # remove expanded dims
    nefc = nefc.reshape((-1,) + nefc.shape[2:])
  if qpos.ndim == 1:  # add batch dim if needed
    qpos.ndim = 2
    qpos = qpos.reshape((1, -1))
  # TODO(btaba): codegen should capture that the same qpos is passed twice!
  return _wp_launch(qpos, ncon, nefc, qpos)


def make_data(m):
  return Data(jp.zeros(m.nq, dtype=float), jp.zeros(1, dtype=int), jp.zeros(1, dtype=int))


class Env:
  def __init__(self, m: Model):
    self._model = m

  def reset(self, rng: jax.Array) -> Data:
    data = make_data(self._model)
    qpos = jax.random.uniform(rng, (self._model.nq,))
    # NB: Any outputs that are not aliased, are malloc'd.
    output_dims = {'qpos': qpos.shape, 'ncon': (1,), 'nefc': (1,)}
    # TODO(btaba): pass named aliases, should make this less error prone.
    jf = ffi.jax_callable(_shim, num_outputs=3, vmap_method='expand_dims',
                          output_dims=output_dims, in_out_argnames={'qpos', 'ncon', 'nefc'})
    qpos, ncon, nefc = jf(data.qpos, data.ncon, data.nefc)
    data = data.replace(qpos=qpos, ncon=ncon, nefc=nefc)
    return data

  def step(self, state: Data):
    # NB: Any outputs that are not aliased, are malloc'd.
    output_dims = {'qpos': state.qpos.shape, 'ncon': (1,), 'nefc': (1,)}
    jf = ffi.jax_callable(_shim, num_outputs=3, vmap_method='expand_dims', output_dims=output_dims,
                          in_out_argnames={'qpos', 'ncon', 'nefc'})
    qpos, ncon, nefc = jf(state.qpos, state.ncon, state.nefc)
    return state.replace(qpos=qpos, ncon=ncon, nefc=nefc)



if __name__ == '__main__':
  wp.clear_kernel_cache()

  m = test_util.load_test_file('pendula.xml')
  rng = jax.random.PRNGKey(0)
  env = Env(m)

  # # Vanilla call.
  # state = jax.jit(env.reset)(rng)
  # state = jax.jit(env.step)(state)
  # assert state.ncon[0] == 2
  # assert state.nefc[0] == -2

  # Vmapped on data.
  batch_size = 8
  rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
  state = jax.jit(jax.vmap(env.reset))(rng)
  with jax.checking_leaks():
    state = jax.jit(jax.vmap(env.step))(state)

  # # # Vmapped on data with in_axes.
  # # in_axes = jax.tree_map(lambda x: 0, dx)
  # # in_axes = in_axes.replace(xquat=None)
  # # dx_batch = dx_batch.replace(xquat=-100 * dx.xquat)
  # # out = jax.jit(jax.vmap(dummy_jax, in_axes=[None, in_axes]))(mx, dx_batch)

  # # # Nested vmap.
  # # dx_batch = jax.vmap(make_data)(rng)
  # # dx_batch2 = jax.tree_map(lambda x: x.reshape((2, 4) + x.shape[1:]), dx_batch)
  # # out = jax.jit(jax.vmap(jax.vmap(dummy_jax, in_axes=[None, 0]), in_axes=[None, 0]))(mx, dx_batch2)


