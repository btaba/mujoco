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
"""Tests for smooth dynamics functions."""

import os
from unittest import mock

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import test_util
import numpy as np


# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr)


class WarpSmoothTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_smooth(self):
    """Tests Warp smooth functions called from MJX."""
    import os
    from unittest import mock

    from absl.testing import absltest
    import jax
    from jax import numpy as jp
    import mujoco
    from mujoco import mjx
    from mujoco.mjx._src import test_util
    import numpy as np

    m = test_util.load_test_file('pendula.xml')
    os.environ["MJX_WARP_ENABLED"] = "true"

    # tell MJX to use sparse mass matrices:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    # randomize mocap
    d.mocap_pos = np.random.random(d.mocap_pos.shape)
    d.mocap_quat = np.random.random(d.mocap_quat.shape)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m, backend_impl='warp')

    # kinematics
    batch_size = 10
    def make_data(rng):
      dx = mjx.make_data(m, backend_impl='warp')
      return dx.replace(qpos=d.qpos.copy())
    dx_batch = jax.vmap(make_data)(jp.arange(batch_size))
    out = jax.jit(jax.vmap(mjx.kinematics, in_axes=(None, 0)))(mx, dx_batch)

    import IPython; IPython.embed(user_ns=dict(globals(), **locals()))

    # dx = mjx.make_data(m, backend_impl='warp')
    # dx = dx.replace(qpos=d.qpos)

    # dx = jax.jit(mjx.kinematics)(mx, dx)
    dx = jax.tree_map(lambda x: x[0], out)
    _assert_attr_eq(d, dx, 'xanchor')
    _assert_attr_eq(d, dx, 'xaxis')
    _assert_attr_eq(d, dx, 'xpos')
    _assert_attr_eq(d, dx, 'xquat')
    # _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
    # _assert_attr_eq(d, dx, 'xipos')
    # _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
    # _assert_attr_eq(d, dx, 'geom_xpos')
    # _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
    # _assert_attr_eq(d, dx, 'site_xpos')
    # _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')


if __name__ == '__main__':
  absltest.main()
