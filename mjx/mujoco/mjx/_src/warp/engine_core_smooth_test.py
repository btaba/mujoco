# Copyright 2024 DeepMind Technologies Limited
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
"""Core smooth function tests."""

import time
from typing import Tuple

from absl.testing import absltest
import jax
import mujoco
from mujoco import mjx
import numpy as np
from mujoco.mjx._src import test_util


class EngineCoreSmoothTest(absltest.TestCase):

  def test_kinematics(self):
    m = test_util.load_test_file('humanoid/10_humanoids.xml')
    mx = mjx.put_model(m)
    dx = mjx.make_data(m)

    dx_x = jax.jit(mjx.kinematics)(mx, dx)
    dx_c = jax.jit(engine_core_smooth.kinematics)(mx, dx)

    np.testing.assert_allclose(dx_x.xanchor, dx_c.xanchor, atol=1e-5)
    np.testing.assert_allclose(dx_x.xaxis, dx_c.xaxis, atol=1e-5)
    np.testing.assert_allclose(dx_x.xmat, dx_c.xmat, atol=1e-5)
    np.testing.assert_allclose(dx_x.xpos, dx_c.xpos, atol=1e-5)
    np.testing.assert_allclose(dx_x.xquat, dx_c.xquat, atol=1e-5)

#   def test_kinematics_batched(self):
#     m = mujoco.MjModel.from_xml_path(  # pylint:disable=disallowed-name
#         "runfile:google3/third_party/mujoco/model/humanoid/humanoid.xml"
#     )
#     mx = mjx.put_model(m)

#     @jax.vmap
#     def make_data(rng):
#       qpos = jax.random.uniform(rng, (m.nq,))
#       dx = mjx.make_data(m).replace(qpos=qpos)
#       return dx

#     dx = make_data(jax.random.split(jax.random.key(0), 128))

#     dx_x = jax.jit(jax.vmap(mjx.kinematics, in_axes=(None, 0)))(mx, dx)
#     dx_c = jax.jit(engine_core_smooth.kinematics)(mx, dx)

#     np.testing.assert_allclose(dx_x.xanchor, dx_c.xanchor, atol=1e-5)
#     np.testing.assert_allclose(dx_x.xaxis, dx_c.xaxis, atol=1e-5)
#     np.testing.assert_allclose(dx_x.xmat, dx_c.xmat, atol=1e-5)
#     np.testing.assert_allclose(dx_x.xpos, dx_c.xpos, atol=1e-5)
#     np.testing.assert_allclose(dx_x.xquat, dx_c.xquat, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
