"""Call mjwarp through mjx."""

import jax
import mujoco
from mujoco import mjx
from mujoco.mjx._src.smooth import kinematics

m = mujoco.MjModel.from_xml_path('./mujoco/mjx/test_data/humanoid/humanoid.xml')
mx = mjx.put_model(m)
dx = mjx.make_data(mx)

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 4)
dx_batch = jax.vmap(lambda rng: dx.replace(qpos=dx.qpos * jax.random.uniform(rng, (1,))))(rng)
dx_batch = jax.jit(jax.vmap(kinematics, in_axes=(None, 0)))(mx, dx_batch)

