"""Run benchmarks."""

import os
import time
from typing import Any, Callable, Sequence, Tuple

from absl import app
from absl import flags
import jax
import mujoco
from mujoco import mjx
from mujoco.mjx.warp import smooth as wp_smooth
from mujoco.mjx.warp import forward as wp_forward
import warp as wp
import mujoco_warp as mjwarp
from warp.jax_experimental import ffi as warp_ffi

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_graph_min_graph_size=1"
)

_MODELFILE = flags.DEFINE_string(
    'modelfile',
    'mujoco/mjx/test_data/humanoid/humanoid.xml',
    'path to model',
)
_FUNCTION = flags.DEFINE_string(
    'function', 'kinematics', 'function to benchmark'
)
_NSTEP = flags.DEFINE_integer('nstep', 1000, 'number of steps per rollout')
_NENV = flags.DEFINE_integer('nenv', 8192, 'number of environments to simulate')
_UNROLL = flags.DEFINE_integer('unroll', 4, 'number of steps to unroll')


def _measure(fn, *args) -> Tuple[float, float]:
  """Reports jit time and op time for a function."""

  beg = time.perf_counter()
  compiled_fn = fn.lower(*args).compile()
  end = time.perf_counter()
  jit_time = end - beg

  times = []

  for i in range(5):
    beg = time.perf_counter()
    result = compiled_fn(*args)
    jax.block_until_ready(result)
    end = time.perf_counter()
    run_time = end - beg
    times.append(run_time)
    print('Measure run: ', i, f', run time: {run_time:.3f}')

  return jit_time, sum(times) / len(times)


def benchmark(
    m: mujoco.MjModel,
    mx: mjx.Model,
    step_fn: Callable[..., Any],
    nstep: int = 1000,
    nenv: int = 8192,
    unroll_steps: int = 4,
) -> Tuple[float, float, int]:
  """Benchmark a model."""

  @jax.vmap
  def init(key):
    qpos = 0.01 * jax.random.normal(key, shape=(m.nq,))
    qvel = 0.01 * jax.random.normal(key, shape=(m.nv,))
    bimpl_ = mx.backend_impl
    d = mjx.make_data(m, backend_impl=bimpl_).replace(
      qpos=qpos, qvel=qvel)
    return d

  key = jax.random.split(jax.random.key(0), nenv)
  d = jax.jit(init)(key)
  jax.block_until_ready(d)

  @jax.jit
  def unroll(d):
    def fn(d, _):
      d = d.replace(qpos=d.qpos + 0.0001 * d.qpos)
      return step_fn(mx, d), None

    return jax.lax.scan(fn, d, None, length=nstep, unroll=unroll_steps)

  jit_time, run_time = _measure(unroll, d)
  steps = nstep * nenv

  return jit_time, run_time, steps


def benchmark_raw_warp(
    m: mujoco.MjModel,
    nstep: int = 1000,
    nenv: int = 8192,
    unroll_steps: int = 4,
    function: str = 'kinematics',
):
  if function != 'kinematics':
    return
  def warp_kinematics(
      qpos_in: wp.array2d(dtype=wp.float32),
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
    wp.copy(d.qpos, qpos_in)
    print('Calling kinematics')
    mjwarp.kinematics(mw, d)
    wp.copy(xpos, d.xpos)
    wp.copy(xquat, d.xquat)
    wp.copy(xmat, d.xmat)
    wp.copy(xipos, d.xipos)
    wp.copy(ximat, d.ximat)
    wp.copy(xanchor, d.xanchor)
    wp.copy(xaxis, d.xaxis)
    wp.copy(geom_xpos, d.geom_xpos)
    wp.copy(geom_xmat, d.geom_xmat)

  def unroll(qpos, xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat):
    def step(carry, _):
      qpos, *_ = carry
      out = warp_kinematics_fn(qpos)
      return (qpos,) + tuple(out), None

    (qpos, xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat), _ = jax.lax.scan(
      step, (qpos, xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat),
      length=nstep, unroll=unroll_steps)

    return qpos, xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat

  output_dims = {
      "xpos": (nenv, m.nbody, 3),
      "xquat": (nenv, m.nbody, 4),
      "xmat": (nenv, m.nbody, 3, 3),
      "xipos": (nenv, m.nbody, 3),
      "ximat": (nenv, m.nbody, 3, 3),
      "xanchor": (nenv, m.njnt, 3),
      "xaxis": (nenv, m.njnt, 3),
      "geom_xpos": (nenv, m.ngeom, 3),
      "geom_xmat": (nenv, m.ngeom, 3, 3),
  }
  warp_kinematics_fn = warp_ffi.jax_callable(
    warp_kinematics,
    num_outputs=9,
    output_dims=output_dims,
  )

  @jax.vmap
  def init(key):
    qpos = 0.01 * jax.random.normal(key, shape=(m.nq,))
    qvel = 0.01 * jax.random.normal(key, shape=(m.nv,))
    d = mjx.make_data(m, backend_impl='jax').replace(
      qpos=qpos, qvel=qvel)
    return d

  key = jax.random.split(jax.random.key(0), nenv)
  dx = jax.jit(init)(key)
  d_ = mujoco.MjData(m)
  mw = mjwarp.put_model(m)
  d = mjwarp.put_data(m, d_, nworld=nenv)

  jax_unroll_fn = jax.jit(unroll)
  jit_time, run_time = _measure(
    jax_unroll_fn,
    dx.qpos, dx.xpos, dx.xquat, dx.xmat, dx.xipos, dx.ximat,
    dx.xanchor, dx.xaxis, dx.geom_xpos, dx.geom_xmat)
  steps = nstep * nenv

  return jit_time, run_time, steps


def _main(_: Sequence[str]):
  """Runs testpeed function."""
  os.environ['MJX_WARP_ENABLED'] = 'true'

  modelfile = _MODELFILE.value
  function_ = _FUNCTION.value
  nstep, nenv, unroll = _NSTEP.value, _NENV.value, _UNROLL.value

  m = mujoco.MjModel.from_xml_path(modelfile)
  
  mx = mjx.put_model(m, backend_impl='jax')
  mw = mjx.put_model(m, backend_impl='warp')

  if function_ == 'kinematics':
    func_warp = jax.vmap(wp_smooth.kinematics, in_axes=(None, 0))
    func_jax = jax.vmap(mjx.kinematics, in_axes=(None, 0))
  elif function_ == 'forward':
    func_warp = jax.vmap(wp_forward.forward, in_axes=(None, 0))
    func_jax = jax.vmap(mjx.forward, in_axes=(None, 0))
  elif function_ == 'step':
    func_warp = jax.vmap(wp_forward.step, in_axes=(None, 0))
    func_jax = jax.vmap(mjx.step, in_axes=(None, 0))
  else:
    raise ValueError(f'Unknown function: {function_}')

  print('testspeed.py:\n')
  print(f' modelfile            : {modelfile}')
  print(f' function             : {function_}')
  print(f' nenv                 : {nenv}')
  print(f' nstep                : {nstep}')
  print(f' timestep             : {m.opt.timestep}')
  print(f' unroll               : {unroll}\n')

  for name, mx_, op in (('JAX WARP FFI', mw, func_warp), ('Pure JAX', mx, func_jax)):
    if op is not None:
      jit_time, run_time, steps = benchmark(
          m, mx_, op, nstep, nenv, unroll
      )

      print(f' {name}:')
      print(f' JIT time             : {jit_time:.2f} s')
      print(f' simulation time      : {run_time:.2f} s')
      print(f' steps per second     : { steps / run_time:,.0f}')
      print(
          f' realtime factor      : { steps * m.opt.timestep / run_time:.2f} x'
      )
      print(f' time per step        : { 1e6 * run_time / steps:.2f} µs\n')

  jit_time, run_time, steps = benchmark_raw_warp(m, nstep, nenv, unroll, function=function_)
  print(f' Pure WARP:')
  print(f' JIT time             : {jit_time:.2f} s')
  print(f' simulation time      : {run_time:.2f} s')
  print(f' steps per second     : { steps / run_time:,.0f}')
  print(
      f' realtime factor      : { steps * m.opt.timestep / run_time:.2f} x'
  )
  print(f' time per step        : { 1e6 * run_time / steps:.2f} µs\n')


def main():
  app.run(_main)


if __name__ == '__main__':
  main()