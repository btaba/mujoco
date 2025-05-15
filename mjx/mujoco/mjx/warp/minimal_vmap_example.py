"""Example."""
import os
import time
from typing import Sequence

from absl import app
from absl import flags
import jax
import jax.numpy as jp
import mujoco_warp as mjwarp
import warp as wp
from warp.jax_experimental import ffi as warp_ffi

import mujoco
from mujoco import mjx


os.environ["XLA_FLAGS"] = (
    "--xla_gpu_graph_min_graph_size=1"
)
_DO_MORE_WORK = flags.DEFINE_bool('do_more_work', False, 'Do more work in the jax_callable.')


def _measure(fn, *args) -> tuple[float, float]:
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


def vmap(m, nenv, nstep, unroll_steps):
  d = mujoco.MjData(m)
  _m = mjwarp.put_model(m)
  _d = mjwarp.put_data(m, d, nworld=nenv)

  def _kinematics_shim(
      qpos: wp.array(dtype=float),
      xpos: wp.array(dtype=wp.vec3),
      xquat: wp.array(dtype=wp.quat),
      xmat: wp.array(dtype=wp.mat33),
      xipos: wp.array(dtype=wp.vec3),
      ximat: wp.array(dtype=wp.mat33),
      xanchor: wp.array(dtype=wp.vec3),
      xaxis: wp.array(dtype=wp.vec3),
      geom_xpos: wp.array(dtype=wp.vec3),
      geom_xmat: wp.array(dtype=wp.mat33),
      site_xpos: wp.array(dtype=wp.vec3),
      site_xmat: wp.array(dtype=wp.mat33),
  ):
    if _DO_MORE_WORK.value:
      a = 0
      for _ in range(1_000_000):
        a += 1
    wp.copy(_d.qpos, qpos)
    mjwarp.kinematics(_m, _d)
    wp.copy(site_xmat, _d.site_xmat)
    wp.copy(site_xpos, _d.site_xpos)
    wp.copy(xanchor, _d.xanchor)
    wp.copy(xaxis, _d.xaxis)
    wp.copy(ximat, _d.ximat)
    wp.copy(xipos, _d.xipos)
    wp.copy(xmat, _d.xmat)
    wp.copy(xpos, _d.xpos)
    wp.copy(xquat, _d.xquat)
    wp.copy(geom_xmat, _d.geom_xmat)
    wp.copy(geom_xpos, _d.geom_xpos)


  def kinematics(dx):
    output_dims = {
        "xpos": (m.nbody, 3),
        "xquat": (m.nbody, 4),
        "xmat": (m.nbody, 3, 3),
        "xipos": (m.nbody, 3),
        "ximat": (m.nbody, 3, 3),
        "xanchor": (m.njnt, 3),
        "xaxis": (m.njnt, 3),
        "geom_xpos": (m.ngeom, 3),
        "geom_xmat": (m.ngeom, 3, 3),
        "site_xpos": (m.nsite, 3),
        "site_xmat": (m.nsite, 3, 3),
    }
    jf = warp_ffi.jax_callable(
        _kinematics_shim,
        num_outputs=11,
        output_dims=output_dims,
        vmap_method='broadcast_all',
        graph_compatible=True,
    )
    out = jf(dx.qpos)
    dx = dx.replace(
        xpos=out[0],
        xquat=out[1],
        xmat=out[2],
        xipos=out[3],
        ximat=out[4],
        xanchor=out[5],
        xaxis=out[6],
        geom_xpos=out[7],
        geom_xmat=out[8],
        site_xpos=out[9],
        site_xmat=out[10],
    )
    return dx

  # Create the data in JAX.
  @jax.vmap
  def init(key):
    qpos = 0.01 * jax.random.normal(key, shape=(m.nq,))
    d = mjx.make_data(m, backend_impl='jax').replace(qpos=qpos)
    return d
  rng = jax.random.key(0)
  key = jax.random.split(rng, nenv)
  dx = jax.jit(init)(key)

  def unroll(*args):
    def step(carry, _):
      dx, rng, *_ = carry
      rng, key = jax.random.split(rng)
      dx = dx.replace(qpos=dx.qpos * jax.random.uniform(key))
      dx = jax.vmap(kinematics)(dx)
      return (dx, rng), None
    out, _ = jax.lax.scan(step, args, length=nstep, unroll=unroll_steps)
    return out

  jax_unroll_fn = jax.jit(unroll)
  jit_time, run_time = _measure(jax_unroll_fn, dx, rng)
  steps = nstep * nenv

  return jit_time, run_time, steps


def pretty_print(name, jit_time, run_time, steps):
  print(f' {name}:')
  print(f' JIT time             : {jit_time:.2f} s')
  print(f' simulation time      : {run_time:.2f} s')
  print(f' steps per second     : { steps / run_time:.0f}')
  print(f' time per step        : { 1e6 * run_time / steps:.2f} µs\n')


def _main(_: Sequence[str]):
  modelfile = 'mujoco/mjx/test_data/humanoid/humanoid.xml'
  nstep, nenv, unroll = 1000, 8192, 4
  m = mujoco.MjModel.from_xml_path(modelfile)

  jit_time, run_time, steps = vmap(m, nenv, nstep, unroll)
  pretty_print('vmap', jit_time, run_time, steps)


if __name__ == '__main__':
  app.run(_main)
