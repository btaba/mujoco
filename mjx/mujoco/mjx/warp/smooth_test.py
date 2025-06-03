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
import warp as wp
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import test_util
import numpy as np
from mujoco.mjx.warp import smooth
from mujoco.mjx.warp import types as mjx_warp_types

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
    wp.clear_kernel_cache()
    np.random.seed(0)

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_kinematics_single(self):
    """Tests Warp smooth from MJX with unbatched data."""
    m = test_util.load_test_file('pendula.xml')

    d = mujoco.MjData(m)
    mx = mjx.put_model(m, backend_impl='warp')

    rng = jax.random.PRNGKey(0)
    dx = mjx.make_data(m, backend_impl='warp')
    rng, key = jax.random.split(rng)
    qpos = jax.random.uniform(key, (m.nq,))
    rng, key1, key2 = jax.random.split(rng, 3)
    mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
    mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
    mocap_quat = math.normalize(mocap_quat)
    dx = dx.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    dx = jax.jit(smooth.kinematics)(mx, dx)

    d.qpos[:] = qpos
    d.mocap_pos[:] = mocap_pos
    d.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(m, d)

    _assert_attr_eq(d, dx, 'xanchor')
    _assert_attr_eq(d, dx, 'xaxis')
    _assert_attr_eq(d, dx, 'xpos')
    _assert_attr_eq(d, dx, 'xquat')
    _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
    _assert_attr_eq(d, dx, 'xipos')
    _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
    _assert_attr_eq(d, dx, 'geom_xpos')
    _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
    _assert_attr_eq(d, dx, 'site_xpos')
    _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_kinematics_batch(self):
    """Tests Warp smooth from MJX with batched data."""
    m = test_util.load_test_file('pendula.xml')

    batch_size = 7
    d = mujoco.MjData(m)
    mx = mjx.put_model(m, backend_impl='warp')

    def make_data(rng):
      dx = mjx.make_data(m, backend_impl='warp')
      rng, key = jax.random.split(rng)
      qpos = jax.random.uniform(key, (m.nq,))
      rng, key1, key2 = jax.random.split(rng, 3)
      mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
      mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
      mocap_quat = math.normalize(mocap_quat)
      return dx.replace(qpos=qpos, mocap_pos=mocap_pos,
                        mocap_quat=mocap_quat)

    rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
    dx_batch = jax.vmap(make_data)(rng)
    out = jax.jit(jax.vmap(smooth.kinematics, in_axes=(None, 0)))(mx, dx_batch)

    for i in range(batch_size):
      dx = jax.tree.map(lambda x: x[i], out)

      d.qpos[:] = dx.qpos
      d.mocap_pos[:] = dx.mocap_pos
      d.mocap_quat[:] = dx.mocap_quat
      mujoco.mj_forward(m, d)

      _assert_attr_eq(d, dx, 'xanchor')
      _assert_attr_eq(d, dx, 'xaxis')
      _assert_attr_eq(d, dx, 'xpos')
      _assert_attr_eq(d, dx, 'xquat')
      _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
      _assert_attr_eq(d, dx, 'xipos')
      _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
      _assert_attr_eq(d, dx, 'geom_xpos')
      _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
      _assert_attr_eq(d, dx, 'site_xpos')
      _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_kinematics_nested_vmap(self):
    """Tests Warp smooth from MJX with batched data."""
    m = test_util.load_test_file('pendula.xml')

    d = mujoco.MjData(m)
    mx = mjx.put_model(m, backend_impl='warp')

    def make_data(rng):
      dx = mjx.make_data(m, backend_impl='warp')
      rng, key = jax.random.split(rng)
      qpos = jax.random.uniform(key, (m.nq,))
      rng, key1, key2 = jax.random.split(rng, 3)
      mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
      mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
      mocap_quat = math.normalize(mocap_quat)
      return dx.replace(qpos=qpos, mocap_pos=mocap_pos,
                        mocap_quat=mocap_quat)

    rng = jax.random.split(jax.random.PRNGKey(0), 8)
    rng = rng.reshape((2, 4, -1))
    dx_batch = jax.vmap(jax.vmap(make_data))(rng)

    out = jax.jit(jax.vmap(jax.vmap(smooth.kinematics, in_axes=(None, 0)), in_axes=(None, 0)))(mx, dx_batch)

    for i in range(2):
      for j in range(4):
        dx = jax.tree.map_with_path(
          lambda path, x: x[i][j]
          if path[-1].name not in mjx_warp_types._DATA_NON_VMAP
          else x, out)

        d.qpos[:] = dx.qpos
        d.mocap_pos[:] = dx.mocap_pos
        d.mocap_quat[:] = dx.mocap_quat
        mujoco.mj_forward(m, d)

        _assert_attr_eq(d, dx, 'xanchor')
        _assert_attr_eq(d, dx, 'xaxis')
        _assert_attr_eq(d, dx, 'xpos')
        _assert_attr_eq(d, dx, 'xquat')
        _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
        _assert_attr_eq(d, dx, 'xipos')
        _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
        _assert_attr_eq(d, dx, 'geom_xpos')
        _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
        _assert_attr_eq(d, dx, 'site_xpos')
        _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')

#   _SPHERE_SPHERE = """
#     <mujoco>
#       <worldbody>
#         <body>
#           <joint type="free"/>
#           <geom pos="0 0 0" size="0.2" type="sphere"/>
#         </body>
#         <body >
#           <joint type="free"/>
#           <geom pos="0 0.3 0" size="0.11" type="sphere"/>
#         </body>
#       </worldbody>
#     </mujoco>
#   """

#   @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
#   def test_fn(self):
#     """Tests Warp smooth from MJX with batched data."""
#     # m = test_util.load_test_file('pendula.xml')
#     m = mujoco.MjModel.from_xml_string(self._SPHERE_SPHERE)
#     d = mujoco.MjData(m)
#     mx = mjx.put_model(m, backend_impl='warp')
#     # import IPython; IPython.embed(user_ns=dict(globals(), **locals()))

#     def make_data(rng):
#       dx = mjx.make_data(m, backend_impl='warp')
#       rng, key = jax.random.split(rng)
#       qpos = jax.random.uniform(key, (m.nq,))
#       rng, key1, key2 = jax.random.split(rng, 3)
#       mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
#       mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
#       mocap_quat = math.normalize(mocap_quat)
#       return dx.replace(qpos=m.qpos0, mocap_pos=mocap_pos,
#                         mocap_quat=mocap_quat)

#     rng = jax.random.split(jax.random.PRNGKey(0), 8)
#     dx_batch = jax.vmap(make_data)(rng)
#     dx_batch = jax.tree.map(lambda x: x.reshape((2, 4) + x.shape[1:]), dx_batch)

#     dx_batch = jax.jit(jax.vmap(jax.vmap(mjx.kinematics, in_axes=(None, 0)), in_axes=(None, 0)))(mx, dx_batch)
#     out = jax.jit(jax.vmap(jax.vmap(smooth.com_pos, in_axes=(None, 0)), in_axes=(None, 0)))(mx, dx_batch)
#     # out = jax.jit(jax.vmap(jax.vmap(collision_driver.collision, in_axes=(None, 0)), in_axes=(None, 0)))(mx, dx_batch)

#     for i in range(2):
#       for j in range(4):
#         dx = jax.tree.map(lambda x: x[i][j], out)

#         d.qpos[:] = dx.qpos
#         d.mocap_pos[:] = dx.mocap_pos
#         d.mocap_quat[:] = dx.mocap_quat
#         mujoco.mj_forward(m, d)

#         _assert_attr_eq(d, dx, 'subtree_com')

# """
# import mujoco_warp as mjwarp
# mw = mjwarp.put_model(m)
# dw = mjwarp.make_data(m, nworld=8)
# with wp.ScopedCapture() as capture:
#   mjwarp.forward(mw, dw)
# wp.capture_launch(capture.graph)
# """

if __name__ == '__main__':
  absltest.main()
