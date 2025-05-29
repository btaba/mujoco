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


@jax.custom_batching.custom_vmap
def step_with_callback(d: Data) -> Data:
  result_shape_dtypes = [
    jax.ShapeDtypeStruct(d.state.shape, d.state.dtype),
    jax.ShapeDtypeStruct(d.buffer.shape, d.buffer.dtype),
  ]
  print(result_shape_dtypes)
  state, buffer = jax.pure_callback(_callback, result_shape_dtypes, d.state, d.buffer, vmap_method='expand_dims')
  return d.replace(state=state, buffer=buffer)


with jax.checking_leaks():
  d = jax.jit(jax.vmap(step_with_callback))(d)


assert d.state.shape[0] == 5
assert d.buffer.shape[0] == 8192


d = make_data(nbatch=8)
d = d.replace(state=d.state.reshape(2, 4))
with jax.checking_leaks():
  d = jax.jit(jax.vmap(jax.vmap(step_with_callback)))(d)



def rollout(data):
  def f(carry, unused_out):
    data = jax.vmap(step_with_callback)(carry)
    return data, None
  data, _ = jax.lax.scan(f, data, length=100, unroll=1)
  return data

jax.jit(rollout(d))


# @step_with_callback.def_vmap
# def step_with_callback_vmap(axis_size, is_batched, data):
#   data_is_batched, = is_batched
#   assert data_is_batched.state
#   assert not data_is_batched.buffer
#   return step_with_callback(data), data_is_batched



import jax
import flax
from jax import numpy as jp
from jax._src.interpreters.batching import register_vmappable


@flax.struct.dataclass
class Data:
  state: jax.Array
  state2: jax.Array
  buffer: jax.Array  # fixed memory buffer
  shape = property(lambda self: self.state2.shape)

def to_elt(cont, _, d, axis):
  return Data(cont(d.state, axis), cont(d.state2, axis), d.buffer)
def from_elt(cont, axis_size, d, axis_dest):
  return Data(cont(axis_size, d.state, axis_dest), cont(axis_size, d.state2, axis_dest), d.buffer)
register_vmappable(Data, int, int, to_elt, from_elt, None)

def make_data(nbatch, nbuffer=8192):
  return Data(jp.zeros(nbatch), jp.zeros((nbatch, 3)), jp.zeros(nbuffer))

def step(d: Data) -> Data:
  state = d.state + 1
  return Data(state, d.state2, d.buffer)


# Working example with register_vmappable.
d = make_data(nbatch=5)
d = jax.jit(jax.vmap(step))(d)
assert d.state.shape[0] == 5
assert d.buffer.shape[0] == 8192

# Now we want to use JAX callbacks (FFI).
def _callback(state, buffer) -> jax.Array:
  print(state.shape, buffer.shape)
  return state + 1, buffer

@jax.custom_batching.custom_vmap
def step_with_callback(d: Data) -> Data:
  result_shape_dtypes = [
    jax.ShapeDtypeStruct(d.state.shape, d.state.dtype),
    jax.ShapeDtypeStruct(d.buffer.shape, d.buffer.dtype),
  ]
  state, buffer = jax.pure_callback(_callback, result_shape_dtypes, d.state, d.buffer, vmap_method='expand_dims')
  return d.replace(state=state, buffer=buffer)


@step_with_callback.def_vmap
def step_with_callback_vmap(axis_size, is_batched, data):
  data_is_batched, = is_batched
  assert data_is_batched.state
  assert not data_is_batched.buffer
  print(data)
  return step_with_callback(data), data_is_batched


d = jax.jit(jax.vmap(step_with_callback, in_axes=(0,)))(d)  # no leaked tracer
assert d.state.shape[0] == 5
assert d.buffer.shape[0] == 8192

d = make_data(nbatch=8)
d = d.replace(state=d.state.reshape((2, 4, -1)), state2=d.state2.reshape((2, 4) + d.state2.shape[1:]))
d = jax.jit(jax.vmap(jax.vmap(step_with_callback)))(d)
