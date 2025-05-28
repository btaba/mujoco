"""Custom vmappable with JAX callback."""
import jax
import flax
from jax import numpy as jp
from jax._src.interpreters.batching import register_vmappable


@flax.struct.dataclass
class Data:
  state: jax.Array
  buffer: jax.Array  # fixed memory buffer
  shape = property(lambda self: self.state.shape)

def to_elt(cont, _, d, axis):
  return Data(cont(d.state, axis), d.buffer)
def from_elt(cont, axis_size, d, axis_dest):
  return Data(cont(axis_size, d.state, axis_dest), d.buffer)
register_vmappable(Data, int, int, to_elt, from_elt, None)

def make_data(nbatch, nbuffer=8192):
  return Data(jp.zeros(nbatch), jp.zeros(nbuffer))

def step(d: Data) -> Data:
  state = d.state + 1
  return Data(state, d.buffer)


# Working example with register_vmappable.
d = make_data(nbatch=5)
d = jax.jit(jax.vmap(step))(d)
assert d.state.shape[0] == 5
assert d.buffer.shape[0] == 8192


# Now we want to use JAX callbacks (FFI).
def _callback(state, buffer) -> jax.Array:
  print(state.shape, buffer.shape)
  return state + 1, buffer

def step_with_callback(d: Data) -> Data:
  result_shape_dtypes = [
    jax.ShapeDtypeStruct(d.state.shape, d.state.dtype),
    jax.ShapeDtypeStruct(d.buffer.shape, d.buffer.dtype),
  ]
  print(result_shape_dtypes)
  state, buffer = jax.pure_callback(_callback, result_shape_dtypes, d.state, d.buffer, vmap_method='expand_dims')
  return d.replace(state=state, buffer=buffer)

d = jax.jit(jax.vmap(step_with_callback))(d)  # we get a leaked tracer