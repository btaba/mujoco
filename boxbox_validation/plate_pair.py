# Per-contact depth analysis of a tilted thin-plate pair under the three
# box-box collider modes, mirroring the ccd-manifold-depth proposal's
# plate_pair analysis.
#
# A plate resting on another at a slight tilt touches along one edge; the
# far corners are separated. Correct per-point depths must reflect that
# asymmetry. Uniform depth stamping (the artifact the ccd-manifold-depth
# proposal fixes in EPA) reports the same depth at every clipped vertex,
# fabricating penetration at the separated corners.
#
# For each contact this prints the collider's dist and the exact signed gap
# at that lateral position, computed analytically for the tilted-plate
# geometry.
import mujoco
import numpy as np

TILT_DEG = 0.05

import math

# near-parallel plates, tilted about y: the contact patch spans the face while the
# true gap varies linearly across it; penetration depths chosen to show uniform
# depth stamping (deep) and fabricated penetration (shallow) in the EPA manifold
_DROP = 0.008*math.cos(math.radians(TILT_DEG)) + 0.08*math.sin(math.radians(TILT_DEG))
POSES = {
    "deep contact (pen 2e-3 at +x edge)": 0.008 + _DROP - 2e-3,
    "grazing contact (pen 1e-4 at +x edge)": 0.008 + _DROP - 1e-4,
}

XML = """
<mujoco>
  <option gravity="0 0 0" boxbox="{mode}"/>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="box" size=".08 .08 .008"/>
    </body>
    <body pos="0 0 {z}" euler="0 """ + str(TILT_DEG) + """ 0">
      <freejoint/>
      <geom type="box" size=".08 .08 .008"/>
    </body>
  </worldbody>
</mujoco>
"""

MODES = {"new": "new", "legacy": "legacy", "gjk": "convex"}


def exact_gap(model, data, pos):
  """Signed gap between the plates along the lower plate's +z, at the
  lateral position of the contact: distance from the lower plate's top face
  to the upper plate's lowest surface point above that (x, y)."""
  # lower plate: top face at z = 0.008 (identity pose)
  # upper plate: transform pos into its frame, project to its bottom face
  p2 = data.geom_xpos[1]
  R2 = data.geom_xmat[1].reshape(3, 3)
  local = R2.T @ (np.asarray(pos) - p2)
  local[2] = -model.geom_size[1][2]  # bottom face of the upper plate
  world = p2 + R2 @ local
  return float(world[2] - model.geom_size[0][2])


def main():
  for pose_name, z in POSES.items():
    print(f"=== {pose_name} ===")
    for mode, attr in MODES.items():
      model = mujoco.MjModel.from_xml_string(XML.format(mode=attr, z=z))
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
      print(f"[{mode}] ncon={data.ncon}")
      fabricated = 0
      for i in range(data.ncon):
        c = data.contact[i]
        gap = exact_gap(model, data, c.pos)
        tag = ""
        if abs(c.dist - gap) > 1e-5:
          tag = f"  <-- depth error {abs(c.dist - gap):.1e}"
        if c.dist < 0 and gap > 1e-6:
          tag = "  <-- fabricated penetration (separated here)"
          fabricated += 1
        print(f"  contact {i}: x={c.pos[0]:+.4f}  dist={c.dist:+.6f}  "
              f"exact_gap={gap:+.6f}{tag}")
      if fabricated:
        print(f"  {fabricated} contact(s) report penetration where the "
              f"surfaces are separated")
      print()


if __name__ == "__main__":
  main()
