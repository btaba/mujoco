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
  print('Warp nefc: ', nefc.ptr)
  print('Warp ncon: ', ncon.ptr)
  wp.launch(_kernel, dim=(qpos_in.shape[0],), inputs=[qpos_in], outputs=[ncon, nefc, qpos_out])


def _shim(
  qpos: wp.array2d(dtype=float),
  ncon: wp.array(dtype=int),
  nefc: wp.array(dtype=int),
):
  if ncon.ndim > 1:
    ncon.ndim -= 1  # remove expanded dims
    ncon = ncon.reshape((-1,) + ncon.shape[2:])
  if nefc.ndim > 1:
    nefc.ndim -= 1  # remove expanded dims
    nefc = nefc.reshape((-1,) + nefc.shape[2:])
  if qpos.ndim == 1:
    qpos.ndim = 2
    qpos = qpos.reshape((1,) + qpos.shape)
  if qpos.ndim > 2:
    qpos.ndim = 2
    qpos = qpos.reshape((-1,) + qpos.shape[-1:])
  return _wp_launch(qpos, ncon, nefc, qpos)


def make_data(m):
  return Data(jp.zeros(m.nq, dtype=float), jp.zeros(1, dtype=int), jp.zeros(1, dtype=int))


def _callable_impl(d: Data) -> Data:
  output_dims = {'qpos': d.qpos.shape, 'ncon': (1,), 'nefc': (1,)}
  jf = ffi.jax_callable(_shim, num_outputs=3, vmap_method=None,
                        output_dims=output_dims, in_out_argnames={'qpos', 'ncon', 'nefc'})
  qpos, ncon, nefc = jf(d.qpos, d.ncon, d.nefc)
  return d.replace(qpos=d.qpos, ncon=ncon, nefc=nefc)  


@jax.custom_batching.custom_vmap
def _callable(d: Data) -> Data:
  if d.qpos.ndim < 2:
    d = d.replace(qpos=d.qpos[None])  
  return _callable_impl(d)

@_callable.def_vmap
def _callable_vmap(axis_size, is_batched, d):
  data_is_batched, = is_batched
  assert data_is_batched.qpos
  assert not data_is_batched.ncon
  print('Data:', d, data_is_batched)
  old_shape = d.qpos.shape
  if d.qpos.ndim > 2:
    extra_dim = d.qpos.ndim - 2
    d = d.replace(qpos=d.qpos.reshape((-1,) + d.qpos.shape[extra_dim + 1:]))
  d = _callable(d)
  d = d.replace(qpos=d.qpos.reshape(old_shape))
  return d, data_is_batched


class Env:
  def __init__(self, m: Model):
    self._model = m

  def reset(self, rng: jax.Array) -> Data:
    data = make_data(self._model)
    qpos = jax.random.uniform(rng, (self._model.nq,))
    return _callable(data.replace(qpos=qpos))

  def step(self, state: Data):
    return _callable(state)



if __name__ == '__main__':
  wp.clear_kernel_cache()

  rng = jax.random.PRNGKey(0)
  env = Env(test_util.load_test_file('pendula.xml'))

  # Vanilla call.
  state = jax.jit(env.reset)(rng)
  state = jax.jit(env.step)(state)
  assert state.ncon[0] == 2
  assert state.nefc[0] == -2

  batch_size = 8
  rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
  state = jax.jit(jax.vmap(env.reset))(rng)
  state = jax.jit(jax.vmap(env.step))(state)

  # Vmapped on data.
  batch_size = 8
  rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
  rng = rng.reshape((2, 4, 2))
  state = jax.jit(jax.vmap(jax.vmap(env.reset)))(rng)
  state = jax.jit(jax.vmap(jax.vmap(env.step)))(state)
