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
import functools
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src import io
from mujoco.mjx._src import render
from mujoco.mjx._src import forward
import mujoco.mjx.warp as mjxw
from mujoco.mjx.warp import test_util as tu
from mujoco.mjx.warp import warp as wp  # pylint: disable=g-importing-member
import numpy as np
import mediapy as media
from mujoco.mjx.third_party import mujoco_warp as mjw

_FORCE_TEST = os.environ.get('MJX_WARP_FORCE_TEST', '0') == '1'


def save_image(pixels, cam_id, width, height, out_fpath='test.png'):
  pixels = pixels[cam_id]
  pixels = pixels.reshape((width, height))
  r = (pixels & 0xFF).astype(np.uint8)
  g = ((pixels >> 8) & 0xFF).astype(np.uint8)
  b = ((pixels >> 16) & 0xFF).astype(np.uint8)
  pixels = np.dstack([r, g, b])
  media.write_image(out_fpath, pixels)



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

    camera_id = 1
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    qpos = jax.random.uniform(key, (m.nq,))

    # JAX
    mx = mjx.put_model(m, impl='warp')
    width, height = 128, 128
    mx = mx.tree_replace({'_impl.render_opt': dataclasses.replace(
        mx._impl.render_opt,
        width=width,
        height=height,
        render_rgb=True,
        render_depth=True,
        use_shadows=True,
        use_textures=True,
    )})
    dx = mjx.make_data(
        m, impl='warp', pixels=width * height, bvh_ngeom=mx._impl.bvh_ngeom,
        enabled_geom_ids=mx._impl.enabled_geom_ids,
        mesh_bounds_size=mx._impl.mesh_bounds_size,
    )
    dx = dx.replace(qpos=qpos)
    dx = jax.jit(forward.forward)(mx, dx)
    dx = jax.jit(render.render)(mx, dx)
    save_image(dx._impl.pixels, camera_id, width, height)

    # Warp
    mw = mjw.put_model(m)
    dw = mjw.make_data(m, nworld=1, nconmax=1_000, njmax=1_000, pixels=width*height, bvh_ngeom=mw.bvh_ngeom)
    dw.qpos = wp.array(qpos[None])
    mjw.forward(mw, dw)
    mjw.build_warp_bvh(mw, dw)
    mjw.render(mw, dw)
    save_image(dw.pixels.numpy()[0], camera_id, width, height)

    # With JAX vmap
    batch_size = 8
    worldids = jp.arange(batch_size)
    dx_batch = jax.vmap(lambda x: mjx.make_data(
        m, impl='warp', pixels=width * height, bvh_ngeom=mx._impl.bvh_ngeom,
        enabled_geom_ids=mx._impl.enabled_geom_ids,
        mesh_bounds_size=mx._impl.mesh_bounds_size,
    ))(worldids)
    qpos = jax.random.uniform(key, (batch_size, m.nq,))
    dx_batch = dx_batch.replace(qpos=qpos)
    dx_batch = jax.jit(jax.vmap(forward.forward, in_axes=(None, 0)))(mx, dx_batch)
    dx_batch = jax.jit(jax.vmap(render.render, in_axes=(None, 0)))(mx, dx_batch)
    save_image(dx_batch._impl.pixels[1], camera_id, width, height)


if __name__ == '__main__':
  absltest.main()
