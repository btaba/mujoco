
import mujoco
from mujoco import mjx
m = mujoco.MjModel.from_xml_path('./mujoco/mjx/test_data/barkour_v0/assets/barkour_v0_mjx.xml')
mx = mjx.put_model(m)
