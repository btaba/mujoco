# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An example integration of MJX with the MuJoCo viewer."""

import time
from typing import Sequence

from absl import app
from absl import flags
import jax
import mujoco
from mujoco import mjx
import mujoco.viewer


#-----------------------------------------------------------------
import time 
#-----------------------------------------------------------------


_MODEL_PATH = flags.DEFINE_string('mjcf', None, 'Path to a MuJoCo MJCF file.',
                                  required=True)
_JIT = flags.DEFINE_bool('jit', False, 'JIT collision step.')


def _main(argv: Sequence[str]) -> None:
  """Launches MuJoCo passive viewer fed by MJX."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  jax.config.update('jax_debug_nans', True)

  print(f'Loading model from: {_MODEL_PATH.value}.')
  m = mujoco.MjModel.from_xml_path(_MODEL_PATH.value)
  d = mujoco.MjData(m)
  mx = mjx.put_model(m)
  dx = mjx.put_data(m, d)

  print(f'Default backend: {jax.default_backend()}')
  print('JIT-compiling the model physics step...')
  start = time.time()
  #-----------------------------------------------------------------
  #  step_fn = jax.jit(mjx.step).lower(mx, dx).compile()
  # step_fn2 = jax.jit(mjx.step_cuda2).lower(mx, dx).compile()
  # step_fn2 = jax.jit(mjx.step_cuda2)
  if _JIT.value:
    step_fn2 = jax.jit(mjx.step_cuda2)
  else:
    step_fn2 = mjx.step_cuda2
  #-----------------------------------------------------------------
  # run only step1 and step3 by jit 
  step_fn1 = jax.jit(mjx.step_cuda1).lower(mx, dx).compile()
  step_fn3 = jax.jit(mjx.step_cuda3).lower(mx, dx).compile()
  #-----------------------------------------------------------------
  elapsed = time.time() - start
  print(f'Compilation took {elapsed}s.')

  with mujoco.viewer.launch_passive(m, d) as v:
    while True:
      start = time.time()

      # TODO(robotics-simulation): recompile when changing disable flags, etc.
      dx = dx.replace(ctrl=d.ctrl, act=d.act, xfrc_applied=d.xfrc_applied)
      dx = dx.replace(qpos=d.qpos, qvel=d.qvel, time=d.time)  # handle resets
      mx = mx.tree_replace({
          'opt.gravity': m.opt.gravity,
          'opt.tolerance': m.opt.tolerance,
          'opt.ls_tolerance': m.opt.ls_tolerance,
          'opt.timestep': m.opt.timestep,
      })

      #-----------------------------------------------------------------
      # dx = step(mx, dx)
      #-----------------------------------------------------------------
      start_time = time.perf_counter()

      dx = step_fn1(mx, dx)
      dx = step_fn2(mx, dx)
      # dx = mjx.step_cuda2(mx, dx) # disable jit 
      dx = step_fn3(mx, dx)
      end_time = time.perf_counter()

      elapsed_time = (end_time - start_time) * 1e4 # milli
      print(f"Elapsed time in ms: {elapsed_time:.2f} ms")
      #-----------------------------------------------------------------

      mjx.get_data_into(d, m, dx)
      v.sync()

      elapsed = time.time() - start
      if elapsed < m.opt.timestep:
        time.sleep(m.opt.timestep - elapsed)


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
