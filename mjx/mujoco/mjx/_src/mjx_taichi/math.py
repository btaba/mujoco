"""Math."""
import taichi as ti
import taichi.math as tm


@ti.func
def mul_quat(u: tm.vec4, v: tm.vec4) -> tm.vec4:
  return tm.vec4([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


@ti.func
def rot_vec_quat(vec: tm.vec3, quat: tm.vec4) -> tm.vec3:
  s, u = quat[0], tm.vec3(quat[1], quat[2], quat[3])
  r = 2.0 * (u.dot(vec) * u) + (s * s - u.dot(u)) * vec
  r = r + 2.0 * s * u.cross(vec)
  return r


@ti.func
def axis_angle_to_quat(axis: tm.vec3, angle: ti.float32) -> tm.vec4:
  s, c = tm.sin(angle * 0.5), tm.cos(angle * 0.5)
  axis = axis * s
  return tm.vec4(c, axis[0], axis[1], axis[2])


@ti.func
def quat_to_mat(quat: tm.vec4) -> tm.mat3:
  vec = tm.vec4(quat[0], quat[1], quat[2], quat[3])
  q = vec.outer_product(vec)

  return tm.mat3(
          q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
          2.0 * (q[1, 2] - q[0, 3]),
          2.0 * (q[1, 3] + q[0, 2]),
          2.0 * (q[1, 2] + q[0, 3]),
          q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
          2.0 * (q[2, 3] - q[0, 1]),
          2.0 * (q[1, 3] - q[0, 2]),
          2.0 * (q[2, 3] + q[0, 1]),
          q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
  )
