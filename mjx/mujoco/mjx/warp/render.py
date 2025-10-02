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

"""DO NOT EDIT. This file is auto-generated."""
import dataclasses
import jax
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi
import mujoco.mjx.third_party.mujoco_warp as mjwarp
from mujoco.mjx.third_party.mujoco_warp._src import types as mjwp_types
import warp as wp


_m = mjwarp.Model(
    **{f.name: None for f in dataclasses.fields(mjwarp.Model) if f.init}
)
_d = mjwarp.Data(
    **{f.name: None for f in dataclasses.fields(mjwarp.Data) if f.init}
)
_o = mjwarp.Option(
    **{f.name: None for f in dataclasses.fields(mjwarp.Option) if f.init}
)
_ro = mjwarp.RenderOptions(
    **{f.name: None for f in dataclasses.fields(mjwarp.RenderOptions) if f.init}
)
_s = mjwarp.Statistic(
    **{f.name: None for f in dataclasses.fields(mjwarp.Statistic) if f.init}
)
_c = mjwarp.Contact(
    **{f.name: None for f in dataclasses.fields(mjwarp.Contact) if f.init}
)
_e = mjwarp.Constraint(
    **{f.name: None for f in dataclasses.fields(mjwarp.Constraint) if f.init}
)


@ffi.format_args_for_warp
def _render_shim(
    # Model
    nworld: int,
    bvh_ngeom: int,
    enabled_geom_ids: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_type: wp.array(dtype=int),
    light_active: wp.array2d(dtype=bool),
    light_castshadow: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mesh_bounds_size: wp.array(dtype=wp.vec3),
    mesh_bvh_ids: wp.array(dtype=wp.uint64),
    mesh_face: wp.array(dtype=wp.vec3i),
    mesh_faceadr: wp.array(dtype=int),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    ncam: int,
    ngeom: int,
    nlight: int,
    tex_adr: wp.array(dtype=int),
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: wp.array(dtype=int),
    tex_width: wp.array(dtype=int),
    render_opt__fov_rad: float,
    render_opt__height: int,
    render_opt__render_depth: bool,
    render_opt__render_rgb: bool,
    render_opt__use_shadows: bool,
    render_opt__use_textures: bool,
    render_opt__width: int,
    # Data
    bvh_id: int,
    cam_xmat: wp.array2d(dtype=wp.mat33),
    cam_xpos: wp.array2d(dtype=wp.vec3),
    depth: wp.array3d(dtype=float),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    group_roots: wp.array(dtype=wp.int32),
    groups: wp.array(dtype=wp.int32),
    light_xdir: wp.array2d(dtype=wp.vec3),
    light_xpos: wp.array2d(dtype=wp.vec3),
    lowers: wp.array(dtype=wp.vec3),
    pixels: wp.array3d(dtype=wp.uint32),
    uppers: wp.array(dtype=wp.vec3),
):
  _m.stat = _s
  _m.opt = _o
  _d.efc = _e
  _d.contact = _c
  _m.render_opt = _ro
  _m.bvh_ngeom = bvh_ngeom
  _m.enabled_geom_ids = enabled_geom_ids
  _m.geom_dataid = geom_dataid
  _m.geom_matid = geom_matid
  _m.geom_rgba = geom_rgba
  _m.geom_size = geom_size
  _m.geom_type = geom_type
  _m.light_active = light_active
  _m.light_castshadow = light_castshadow
  _m.light_type = light_type
  _m.mat_rgba = mat_rgba
  _m.mat_texid = mat_texid
  _m.mat_texrepeat = mat_texrepeat
  _m.mesh_bounds_size = mesh_bounds_size
  _m.mesh_bvh_ids = mesh_bvh_ids
  _m.mesh_face = mesh_face
  _m.mesh_faceadr = mesh_faceadr
  _m.mesh_texcoord = mesh_texcoord
  _m.mesh_texcoord_offsets = mesh_texcoord_offsets
  _m.ncam = ncam
  _m.ngeom = ngeom
  _m.nlight = nlight
  _m.render_opt.fov_rad = render_opt__fov_rad
  _m.render_opt.height = render_opt__height
  _m.render_opt.render_depth = render_opt__render_depth
  _m.render_opt.render_rgb = render_opt__render_rgb
  _m.render_opt.use_shadows = render_opt__use_shadows
  _m.render_opt.use_textures = render_opt__use_textures
  _m.render_opt.width = render_opt__width
  _m.tex_adr = tex_adr
  _m.tex_data = tex_data
  _m.tex_height = tex_height
  _m.tex_width = tex_width
  _d.bvh_id = bvh_id
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.depth = depth
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.group_roots = group_roots
  _d.groups = groups
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  _d.lowers = lowers
  _d.pixels = pixels
  _d.uppers = uppers
  _d.nworld = nworld
  mjwarp.render(_m, _d)


def _render_jax_impl(m: types.Model, d: types.Data):
  output_dims = {
      'cam_xmat': d.cam_xmat.shape,
      'cam_xpos': d.cam_xpos.shape,
      'depth': d._impl.depth.shape,
      'geom_xmat': d.geom_xmat.shape,
      'geom_xpos': d.geom_xpos.shape,
      'group_roots': d._impl.group_roots.shape,
      'groups': d._impl.groups.shape,
      'light_xdir': d._impl.light_xdir.shape,
      'light_xpos': d._impl.light_xpos.shape,
      'lowers': d._impl.lowers.shape,
      'pixels': d._impl.pixels.shape,
      'uppers': d._impl.uppers.shape,
  }
  jf = ffi.jax_callable_variadic_tuple(
      _render_shim,
      num_outputs=12,
      output_dims=output_dims,
      vmap_method=None,
      in_out_argnames={
          'cam_xmat',
          'cam_xpos',
          'depth',
          'geom_xmat',
          'geom_xpos',
          'group_roots',
          'groups',
          'light_xdir',
          'light_xpos',
          'lowers',
          'pixels',
          'uppers',
      },
  )
  out = jf(
      d.qpos.shape[0],
      m._impl.bvh_ngeom,
      m._impl.enabled_geom_ids,
      m.geom_dataid,
      m.geom_matid,
      m.geom_rgba,
      m.geom_size,
      m.geom_type,
      m._impl.light_active,
      m.light_castshadow,
      m.light_type,
      m.mat_rgba,
      m.mat_texid,
      m._impl.mat_texrepeat,
      m._impl.mesh_bounds_size,
      m._impl.mesh_bvh_ids,
      m.mesh_face,
      m.mesh_faceadr,
      m.mesh_texcoord,
      m._impl.mesh_texcoord_offsets,
      m.ncam,
      m.ngeom,
      m.nlight,
      m.tex_adr,
      m.tex_data,
      m.tex_height,
      m.tex_width,
      m._impl.render_opt.fov_rad,
      m._impl.render_opt.height,
      m._impl.render_opt.render_depth,
      m._impl.render_opt.render_rgb,
      m._impl.render_opt.use_shadows,
      m._impl.render_opt.use_textures,
      m._impl.render_opt.width,
      d._impl.bvh_id,
      d.cam_xmat,
      d.cam_xpos,
      d._impl.depth,
      d.geom_xmat,
      d.geom_xpos,
      d._impl.group_roots,
      d._impl.groups,
      d._impl.light_xdir,
      d._impl.light_xpos,
      d._impl.lowers,
      d._impl.pixels,
      d._impl.uppers,
  )
  d = d.tree_replace({
      'cam_xmat': out[0],
      'cam_xpos': out[1],
      '_impl.depth': out[2],
      'geom_xmat': out[3],
      'geom_xpos': out[4],
      '_impl.group_roots': out[5],
      '_impl.groups': out[6],
      '_impl.light_xdir': out[7],
      '_impl.light_xpos': out[8],
      '_impl.lowers': out[9],
      '_impl.pixels': out[10],
      '_impl.uppers': out[11],
  })
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def render(m: types.Model, d: types.Data):
  return _render_jax_impl(m, d)


@render.def_vmap
@ffi.marshal_custom_vmap
def render_vmap(unused_axis_size, is_batched, m, d):
  d = render(m, d)
  return d, is_batched[1]
