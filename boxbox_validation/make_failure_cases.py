# Generates MJCF models with keyframes posing a box pair (and an identical
# box-mesh pair, offset in +y) at configurations where GJK/EPA's reported
# depth deviates from the exact penetration depth by many orders of magnitude
# more than the box-box collider's.
#
# The poses were found by randomized search through the full pipeline
# (search_gjk_failures.py), scored against the exact depth (max support
# separation over the 15 candidate axes, which attains the true value for
# box-box penetration). The pattern across ~30k contacts: box-box matches
# the exact depth to machine precision (median error 1e-17) on >90% of
# contacts, while EPA carries its iterative tolerance -- typically 1e-9,
# up to ~1e-5 depending on pose. EPA's exact error at a given pose varies
# with absolute position, so the verifier's numbers are the authority.
# Run verify_failure_cases.py to print them.
import os

MESH_OFFSET_Y = 0.6

# (size1, size2, pos2, quat1, quat2, note)
CASES = [
    ((0.018, 0.038, 0.047), (0.026, 0.0014, 0.008),
     (-0.02680797842451752, 0.03042638084962034, 0.00499093437264194),
     (0.3442297784932157, 0.38419095993778196, 0.050988469265927795,
      0.8551627575511663),
     (0.36061702670907897, 0.725243379596019, -0.4724918222125669,
      -0.34746061408726026),
     "anisotropic boxes: EPA error pose-dependent (1e-10..6e-5), box-box exact to 9e-18"),
    ((0.018, 0.038, 0.047), (0.026, 0.0014, 0.008),
     (0.017156021190878595, 0.02613153680265145, -0.008374258475911963),
     (-0.549565061908584, -0.2938118039042643, 0.035208375027483994,
      -0.7812894706469943),
     (-0.3268457642238624, -0.5907999141080185, -0.6283058938227173,
      0.3864699363294817),
     "anisotropic boxes: EPA off by ~1e-5, box-box exact to 9e-18"),
    ((0.08, 0.08, 0.004), (0.06, 0.06, 0.006),
     (0.11780650774183882, 0.01793022277628169, -0.11507185266968654),
     (0.8783624402602853, -0.3647796272348773, 0.2227610685910456,
      -0.21399241438554814),
     (-0.09168443273868333, -0.4382220279162731, 0.24532003802230173,
      -0.8598683026993562),
     "thin slabs: EPA at its 4e-9 tolerance floor, box-box exact to 5e-17"),
    ((0.05, 0.05, 0.05), (0.05, 0.05, 0.05),
     (0.07941605777031256, 0.012804067033715776, 0.08694395726953959),
     (-0.6984632289347097, -0.419531831388011, -0.45185191600467484,
      -0.36327951536141584),
     (0.4512904559565059, 0.824129072755829, -0.05629499174977122,
      -0.33760786380811686),
     "cubes: EPA at its 2e-9 tolerance floor, box-box exact to 2e-17"),
]

TEMPLATE = """<mujoco model="boxbox vs gjk case {idx}">
  <!-- {note}.
       Box pair collides via mjc_BoxBox; the identical box-mesh pair (offset
       +y) collides via GJK/EPA. Load the keyframe and compare each pair's
       deepest contact against the true penetration depth
       (verify_failure_cases.py prints all three). -->
  <option gravity="0 0 0"/>

  <asset>
    <mesh name="m1" scale="{s1x} {s1y} {s1z}"
      vertex="-1 -1 -1  1 -1 -1  1 1 -1  1 1 1  1 -1 1  -1 1 -1  -1 1 1  -1 -1 1"/>
    <mesh name="m2" scale="{s2x} {s2y} {s2z}"
      vertex="-1 -1 -1  1 -1 -1  1 1 -1  1 1 1  1 -1 1  -1 1 -1  -1 1 1  -1 -1 1"/>
  </asset>

  <worldbody>
    <body name="boxA">
      <freejoint/>
      <geom type="box" size="{s1x} {s1y} {s1z}" contype="1" conaffinity="1" rgba=".2 .6 .9 .7"/>
    </body>
    <body name="boxB">
      <freejoint/>
      <geom type="box" size="{s2x} {s2y} {s2z}" contype="1" conaffinity="1" rgba=".9 .5 .2 .7"/>
    </body>
    <body name="meshA" pos="0 {offy} 0">
      <freejoint/>
      <geom type="mesh" mesh="m1" contype="2" conaffinity="2" rgba=".2 .6 .9 .7"/>
    </body>
    <body name="meshB" pos="0 {offy} 0">
      <freejoint/>
      <geom type="mesh" mesh="m2" contype="2" conaffinity="2" rgba=".9 .5 .2 .7"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="failure"
         qpos="0 0 0  {q1}
               {p2}  {q2}
               0 {offy} 0  {q1}
               {p2y}  {q2}"/>
  </keyframe>
</mujoco>
"""


def fmt(vals):
  return " ".join(f"{v:.17g}" for v in vals)


def main():
  outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "failure_cases")
  os.makedirs(outdir, exist_ok=True)
  for i, (s1, s2, pos2, q1, q2, note) in enumerate(CASES, 1):
    p2y = (pos2[0], pos2[1] + MESH_OFFSET_Y, pos2[2])
    xml = TEMPLATE.format(
        idx=i, note=note,
        s1x=s1[0], s1y=s1[1], s1z=s1[2],
        s2x=s2[0], s2y=s2[1], s2z=s2[2],
        offy=MESH_OFFSET_Y,
        q1=fmt(q1), q2=fmt(q2), p2=fmt(pos2), p2y=fmt(p2y))
    path = os.path.join(outdir, f"case{i}.xml")
    with open(path, "w") as f:
      f.write(xml)
    print("wrote", path)


if __name__ == "__main__":
  main()
