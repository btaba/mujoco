# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for collision functions."""

import os
from unittest import mock

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
import warp as wp
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import test_util
import numpy as np
from mujoco.mjx.warp import smooth
from mujoco.mjx.warp import types as mjx_warp_types
from mujoco.mjx.warp import collision_driver
# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr)


class WarpCollisionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.clear_kernel_cache()
    np.random.seed(0)

  _SPHERE_SPHERE = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="free"/>
          <geom pos="0 0 0" size="0.2" type="sphere"/>
        </body>
        <body >
          <joint type="free"/>
          <geom pos="0 0.3 0" size="0.11" type="sphere"/>
        </body>
      </worldbody>
    </mujoco>
  """

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_fn(self):
    """Tests Warp smooth from MJX with batched data."""
    m = mujoco.MjModel.from_xml_string(self._SPHERE_SPHERE)
    d = mujoco.MjData(m)
    mx = mjx.put_model(m, backend_impl='warp')

    def make_data(rng):
      dx = mjx.make_data(m, backend_impl='warp')
      rng, key = jax.random.split(rng)
      qpos = jax.random.uniform(key, (m.nq,), minval=-0.01, maxval=0.01)
      return dx.replace(qpos=qpos,)

    rng = jax.random.split(jax.random.PRNGKey(0), 8)
    rng = rng.reshape((2, 4, -1))
    dx_batch = jax.vmap(jax.vmap(make_data))(rng)

    dx_batch = jax.jit(jax.vmap(jax.vmap(mjx.kinematics, in_axes=(None, 0)), in_axes=(None, 0)))(mx, dx_batch)
    out = jax.jit(jax.vmap(jax.vmap(collision_driver.collision, in_axes=(None, 0)), in_axes=(None, 0)))(mx, dx_batch)

    idx = 0
    for i in range(2):
      for j in range(4):
        dx = jax.tree.map_with_path(
          lambda path, x: x[i][j]
          if path[-1].name not in mjx_warp_types._DATA_NON_VMAP
          else x, out)

        d.qpos[:] = dx.qpos
        mujoco.mj_forward(m, d)

        if not len(d.contact.pos):
          continue

        np.testing.assert_allclose(out._impl.contact__worldid[idx], i * 4 + j)
        np.testing.assert_allclose(out._impl.contact__dist[idx], d.contact.dist[0], atol=1e-4)
        idx += 1


if __name__ == '__main__':
  absltest.main()
