"""Minimal test for pmap and MJX-Warp."""
import mujoco
from mujoco import mjx
import jax
from jax import numpy as jp
from mujoco.mjx.third_party import mujoco_warp
import warp as wp


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="floor" pos="0 0 0">
      <geom name="floor" type="plane" size="1 1 0.1" rgba=".8 .8 .8 1"/>
    </body>
    <body name="sphere" pos="0 0 1">
      <joint type="free"/>
      <geom name="sphere" type="sphere" size="0.1" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def mjx_warp():
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl='warp')

  def make_data(rng):
    dx = mjx.make_data(m, impl='warp', nconmax=10, njmax=10)
    dx = dx.replace(qpos=jax.random.uniform(rng, mx.nq))
    return dx

  device_count = jax.local_device_count()
  assert device_count > 1
  batch_size = 16

  keys = jax.random.split(jax.random.PRNGKey(0), device_count * batch_size)
  keys = keys.reshape((device_count, batch_size, -1)) 
  dx_batch = jax.pmap(jax.vmap(make_data))(keys)
  assert dx_batch.qpos.shape == (device_count, batch_size, m.nq)

  out = jax.pmap(jax.vmap(mjx.step, in_axes=(None, 0)), in_axes=(None, 0))(mx, dx_batch)


def mjwarp():
  m = mujoco.MjModel.from_xml_string(xml)

  mws, dws, graphs = [], [], {}
  for device in wp.get_cuda_devices():
    with wp.ScopedDevice(device):
      mw = mujoco_warp.put_model(m)
      dw = mujoco_warp.make_data(m, nconmax=100, njmax=100, nworld=16)

      with wp.ScopedCapture(device) as capture:
        mujoco_warp.step(mw, dw)

      mws.append(mw)
      dws.append(dw)
      graphs[str(device)] = capture.graph

  for device in wp.get_cuda_devices():
    wp.capture_launch(graphs[str(device)])

if __name__ == '__main__':
  mjwarp()
  mjx_warp()
