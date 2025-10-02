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
"""Tests for codegen'd smooth functions."""

import dataclasses
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from mujoco import mjx
from mujoco.mjx._src import io
from mujoco.mjx._src import render
import mujoco.mjx.warp as mjxw
from mujoco.mjx.warp import test_util as tu
from mujoco.mjx.warp import warp as wp  # pylint: disable=g-importing-member
import numpy as np

_FORCE_TEST = os.environ.get('MJX_WARP_FORCE_TEST', '0') == '1'


class RenderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if mjxw.WARP_INSTALLED:
      # self.tempdir = tempfile.TemporaryDirectory()
      tempdir = '/tmp/wp_kernel_cache_dir_RenderTest'
      # wp.config.kernel_cache_dir = self.tempdir.name
      wp.config.kernel_cache_dir = tempdir
    np.random.seed(0)

  def tearDown(self):
    super().tearDown()
    if hasattr(self, 'tempdir'):
      self.tempdir.cleanup()

  def test_render(self):
    """Tests render."""
    if not _FORCE_TEST:
      if not mjxw.WARP_INSTALLED:
        self.skipTest('Warp not installed.')
      if not io.has_cuda_gpu_device():
        self.skipTest('No CUDA GPU device available.')

    m = tu.load_test_file('humanoid/humanoid.xml')
    mx = mjx.put_model(m, impl='warp')

    width, height = 64, 64
    mx = mx.tree_replace({'_impl.render_opt': dataclasses.replace(
        mx._impl.render_opt,
        width=width,
        height=height,
        render_rgb=True,
        render_depth=True,
        use_shadows=True,
        use_textures=True,
    )})

    rng = jax.random.PRNGKey(0)
    dx = mjx.make_data(
        m, impl='warp', pixels=width * height, bvh_ngeom=mx._impl.bvh_ngeom,
        enabled_geom_ids=mx._impl.enabled_geom_ids,
        mesh_bounds_size=mx._impl.mesh_bounds_size,
    )
    rng, key = jax.random.split(rng)
    qpos = jax.random.uniform(key, (m.nq,))
    dx = dx.replace(qpos=qpos)

    dx = jax.jit(render.render)(mx, dx)
    import IPython; IPython.embed(user_ns=dict(globals(), **locals()))


    # batch_size = 8
    # worldids = jp.arange(batch_size)
    # dx_batch = jax.vmap(functools.partial(tu.make_data, m))(worldids)

  # def test_kinematics_vmap(self):
  #   """Tests kinematics with batched data."""
  #   if not mjxw.WARP_INSTALLED:
  #     self.skipTest('Warp not installed.')
  #   if not io.has_cuda_gpu_device():
  #     self.skipTest('No CUDA GPU device available.')

  #   m = tu.load_test_file('pendula.xml')

  #   batch_size = 7
  #   d = mujoco.MjData(m)
  #   mx = mjx.put_model(m, impl='warp')

  #   worldids = jp.arange(batch_size)
  #   dx_batch = jax.vmap(functools.partial(tu.make_data, m))(worldids)
  #   fields = ('xanchor', 'xaxis', 'xpos', 'xquat', 'xmat', 'xipos', 'ximat',
  #             'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat')  # fmt: skip
  #   for f in fields:
  #     dx_batch = dx_batch.replace(**{f: jp.zeros_like(getattr(dx_batch, f))})

  #   dx_batch = jax.jit(jax.vmap(smooth.kinematics, in_axes=(None, 0)))(
  #       mx, dx_batch
  #   )

  #   for i in range(batch_size):
  #     dx = dx_batch[i]

  #     d.qpos[:] = dx.qpos
  #     d.mocap_pos[:] = dx.mocap_pos
  #     d.mocap_quat[:] = dx.mocap_quat
  #     mujoco.mj_forward(m, d)

  #     tu.assert_attr_eq(d, dx, 'xanchor')
  #     tu.assert_attr_eq(d, dx, 'xaxis')
  #     tu.assert_attr_eq(d, dx, 'xpos')
  #     tu.assert_attr_eq(d, dx, 'xquat')
  #     tu.assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
  #     tu.assert_attr_eq(d, dx, 'xipos')
  #     tu.assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
  #     tu.assert_attr_eq(d, dx, 'geom_xpos')
  #     tu.assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
  #     tu.assert_attr_eq(d, dx, 'site_xpos')
  #     tu.assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')


if __name__ == '__main__':
  absltest.main()
