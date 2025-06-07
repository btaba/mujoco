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
"""Tests for forward functions."""

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


# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr)


class WarpForwardTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.clear_kernel_cache()
    np.random.seed(0)

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_forward_single(self):
    """Tests Warp forward from MJX with unbatched data."""
    # m = test_util.load_test_file('pendula.xml')
    # TODO(btaba): graph capture is failing with pendula
    m = test_util.load_test_file('humanoid/humanoid.xml')
    m.opt.iterations = 10
    m.opt.ls_iterations = 10

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qpos0 = d.qpos

    mx = mjx.put_model(m, backend_impl='warp')

    rng = jax.random.PRNGKey(0)
    dx = mjx.make_data(m, backend_impl='warp')
    rng, key = jax.random.split(rng)
    qpos = qpos0 + jax.random.uniform(key, (m.nq,), minval=-0.05, maxval=0.05)
    rng, key1, key2 = jax.random.split(rng, 3)
    mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
    mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
    mocap_quat = math.normalize(mocap_quat)
    dx = dx.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    dx = jax.jit(mjx.forward)(mx, dx)

    d.qpos[:] = qpos
    d.mocap_pos[:] = mocap_pos
    d.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(m, d)

    _assert_attr_eq(d, dx, 'geom_xpos')
    _assert_attr_eq(d, dx, 'subtree_com')
    _assert_attr_eq(d, dx, 'act_dot')
    _assert_attr_eq(d, dx, 'qfrc_actuator')
    _assert_attr_eq(d, dx, 'actuator_force')
    _assert_attr_eq(d, dx, 'qfrc_smooth')
    _assert_attr_eq(d, dx, 'qacc_smooth')
    _assert_attr_eq(d, dx, 'qacc')

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_forward_batch(self):
    """Tests Warp forward from MJX with batched data."""
    m = test_util.load_test_file('humanoid/humanoid.xml')
    m.opt.iterations = 10
    m.opt.ls_iterations

    batch_size = 7
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qpos0 = d.qpos

    mx = mjx.put_model(m, backend_impl='warp')

    def make_data(rng):
      dx = mjx.make_data(m, backend_impl='warp')
      rng, key = jax.random.split(rng)
      qpos = qpos0 + jax.random.uniform(key, (m.nq,), minval=-0.05, maxval=0.05)
      rng, key1, key2 = jax.random.split(rng, 3)
      mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
      mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
      mocap_quat = math.normalize(mocap_quat)
      return dx.replace(qpos=qpos, mocap_pos=mocap_pos,
                        mocap_quat=mocap_quat)

    rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
    dx_batch = jax.vmap(make_data)(rng)
    out = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))(mx, dx_batch)

    for i in range(batch_size):
      dx = jax.tree.map(lambda x: x[i], out)

      d.qpos[:] = dx.qpos
      d.mocap_pos[:] = dx.mocap_pos
      d.mocap_quat[:] = dx.mocap_quat
      mujoco.mj_forward(m, d)

      _assert_attr_eq(d, dx, 'geom_xpos')
      _assert_attr_eq(d, dx, 'subtree_com')
      _assert_attr_eq(d, dx, 'act_dot')
      _assert_attr_eq(d, dx, 'qfrc_actuator')
      _assert_attr_eq(d, dx, 'actuator_force')
      _assert_attr_eq(d, dx, 'qfrc_smooth')
      _assert_attr_eq(d, dx, 'qacc_smooth')


class WarpStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.clear_kernel_cache()
    np.random.seed(0)

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_step_single(self):
    """Tests Warp step from MJX with unbatched data."""
    # m = test_util.load_test_file('pendula.xml')
    # TODO(btaba): graph capture is failing with pendula
    m = test_util.load_test_file('humanoid/humanoid.xml')
    m.opt.iterations = 10
    m.opt.ls_iterations = 10

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qpos0 = d.qpos

    mx = mjx.put_model(m, backend_impl='warp')

    rng = jax.random.PRNGKey(0)
    dx = mjx.make_data(m, backend_impl='warp')
    rng, key = jax.random.split(rng)
    qpos = qpos0 + jax.random.uniform(key, (m.nq,), minval=-0.05, maxval=0.05)
    rng, key1, key2 = jax.random.split(rng, 3)
    mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
    mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
    mocap_quat = math.normalize(mocap_quat)
    dx = dx.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    dx = jax.jit(mjx.step)(mx, dx)

    d.qpos[:] = qpos
    d.mocap_pos[:] = mocap_pos
    d.mocap_quat[:] = mocap_quat
    mujoco.mj_step(m, d)

    _assert_attr_eq(d, dx, 'geom_xpos')
    _assert_attr_eq(d, dx, 'subtree_com')
    _assert_attr_eq(d, dx, 'act_dot')
    _assert_attr_eq(d, dx, 'qfrc_actuator')
    _assert_attr_eq(d, dx, 'actuator_force')
    _assert_attr_eq(d, dx, 'qfrc_smooth')
    _assert_attr_eq(d, dx, 'qacc_smooth')
    _assert_attr_eq(d, dx, 'qvel')
    _assert_attr_eq(d, dx, 'qpos')
    _assert_attr_eq(d, dx, 'time')

  @mock.patch.dict(os.environ, {'MJX_WARP_ENABLED': 'true'})
  def test_step_batch(self):
    """Tests Warp step from MJX with batched data."""
    m = test_util.load_test_file('humanoid/humanoid.xml')
    m.opt.iterations = 20
    m.opt.ls_iterations = 10

    batch_size = 7
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qpos0 = d.qpos

    mx = mjx.put_model(m, backend_impl='warp')
    mx = mx.tree_replace({'opt.ls_parallel': False})

    def make_data(rng):
      dx = mjx.make_data(m, backend_impl='warp')
      rng, key = jax.random.split(rng)
      qpos = qpos0 + jax.random.uniform(key, (m.nq,), minval=-0.02, maxval=0.02)
      rng, key1, key2 = jax.random.split(rng, 3)
      mocap_pos = jax.random.normal(key1, (m.nmocap, 3))
      mocap_quat = jax.random.normal(key2, (m.nmocap, 4))
      mocap_quat = math.normalize(mocap_quat)
      return dx.replace(qpos=qpos, mocap_pos=mocap_pos,
                        mocap_quat=mocap_quat)

    rng = jax.random.split(jax.random.PRNGKey(0), batch_size)
    dx_batch = jax.vmap(make_data)(rng)
    out = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))(mx, dx_batch)

    for i in range(batch_size):
      dx_old = make_data(rng[i])
      d = mujoco.MjData(m)
      d.qpos[:] = dx_old.qpos
      d.qvel[:] = dx_old.qvel
      d.mocap_pos[:] = dx_old.mocap_pos
      d.mocap_quat[:] = dx_old.mocap_quat
      d.qacc_warmstart[:] = dx_old.qacc_warmstart
      mujoco.mj_step(m, d)

      dx = jax.tree.map(lambda x: x[i], out)
      _assert_attr_eq(d, dx, 'geom_xpos')
      _assert_attr_eq(d, dx, 'subtree_com')
      _assert_attr_eq(d, dx, 'act_dot')
      _assert_attr_eq(d, dx, 'qfrc_actuator')
      _assert_attr_eq(d, dx, 'actuator_force')
      _assert_attr_eq(d, dx, 'qfrc_smooth')
      _assert_attr_eq(d, dx, 'qacc_smooth')
      # TODO(btaba): re-enable test for qvel/qpos after some debugging or re-import
      # _assert_attr_eq(d, dx, 'qvel')
      # _assert_attr_eq(d, dx, 'qpos')
      _assert_attr_eq(d, dx, 'time')


if __name__ == '__main__':
  absltest.main()
