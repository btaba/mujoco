"""Run benchmarks."""

import os
import time
from typing import Any, Callable, Sequence, Tuple

from absl import app
from absl import flags
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx.warp import smooth as wp_smooth


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

  for _ in range(5):
    beg = time.perf_counter()
    result = compiled_fn(*args)
    jax.block_until_ready(result)
    end = time.perf_counter()
    run_time = end - beg
    times.append(run_time)

  return jit_time, sum(times) / len(times)


def benchmark(
    m: mujoco.MjModel,
    mx: mjx.Model,
    step_fn: Callable[..., Any],
    nstep: int = 1000,
    nenv: int = 8192,
    unroll_steps: int = 4,
    function: str = 'kinematics',
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
      # ensure real work - avoid caching
      if function == 'kinematics':
        hash_val = jp.sum(d.xpos) + jp.sum(d.xquat)
        hash_val += jp.sum(d.xanchor) + jp.sum(d.xaxis)
        hash_val *= 1e-6
        d = d.replace(qpos=d.qpos + hash_val)
      return step_fn(mx, d), None

    return jax.lax.scan(fn, d, None, length=nstep, unroll=unroll_steps)

  jit_time, run_time = _measure(unroll, d)
  steps = nstep * nenv

  return jit_time, run_time, steps


def _main(_: Sequence[str]):
  """Runs testpeed function."""
  os.environ['MJX_WARP_ENABLED'] = 'true'

  modelfile = _MODELFILE.value
  function = _FUNCTION.value
  nstep, nenv, unroll = _NSTEP.value, _NENV.value, _UNROLL.value

  m = mujoco.MjModel.from_xml_path(modelfile)
  
  mx = mjx.put_model(m, backend_impl='jax')
  mw = mjx.put_model(m, backend_impl='warp')

  if function == 'kinematics':
    func_warp = jax.vmap(wp_smooth.kinematics, in_axes=(None, 0))
    func_jax = jax.vmap(mjx.kinematics, in_axes=(None, 0))
  else:
    raise ValueError(f'Unknown function: {function}')

  print('testspeed.py:\n')
  print(f' modelfile            : {modelfile}')
  print(f' function             : {function}')
  print(f' nenv                 : {nenv}')
  print(f' nstep                : {nstep}')
  print(f' timestep             : {m.opt.timestep}')
  print(f' unroll               : {unroll}\n')

  for name, mx_, op in (('WARP', mw, func_warp), ('Pure JAX', mx, func_jax)):
    if op is not None:
      jit_time, run_time, steps = benchmark(
          m, mx_, op, nstep, nenv, unroll, function=function
      )

      print(f' {name}:')
      print(f' JIT time             : {jit_time:.2f} s')
      print(f' simulation time      : {run_time:.2f} s')
      print(f' steps per second     : { steps / run_time:.0f}')
      print(
          f' realtime factor      : { steps * m.opt.timestep / run_time:.2f} x'
      )
      print(f' time per step        : { 1e6 * run_time / steps:.2f} µs\n')


def main():
  app.run(_main)


if __name__ == '__main__':
  main()