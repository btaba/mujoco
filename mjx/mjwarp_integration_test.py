"""Call mjwarp through mjx."""

import jax
import mujoco
from mujoco import mjx
from mujoco.mjx._src.smooth import kinematics
import warp as wp

wp.config.mode = "debug"
wp.config.print_launches = True
wp.config.verbose = True
wp.config.verbose_warnings = True
wp.config.verify_cuda = True
wp.config.verify_fp = True

m = mujoco.MjModel.from_xml_path('./mujoco/mjx/test_data/humanoid/humanoid.xml')
mx = mjx.put_model(m)
dx = mjx.make_data(mx)

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 4)
dx_batch = jax.vmap(lambda rng: dx.replace(qpos=dx.qpos * jax.random.uniform(rng, (1,))))(rng)
print(1, jax.vmap(kinematics, in_axes=(None, 0))(mx, dx_batch).xpos)
print(2, jax.vmap(kinematics, in_axes=(None, 0))(mx, dx_batch).xpos)
print(3, jax.jit(jax.vmap(kinematics, in_axes=(None, 0)))(mx, dx_batch).xpos)
#print(3, jax.vmap(kinematics, in_axes=(None, 0))(mx, dx_batch).xpos)

