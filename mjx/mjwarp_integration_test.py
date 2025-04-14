
import mujoco
from mujoco import mjx
m = mujoco.MjModel.from_xml_path('./mujoco/mjx/test_data/humanoid/humanoid.xml')
mx = mjx.put_model(m)


