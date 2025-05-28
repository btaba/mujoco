import dataclasses
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi_helper
import mujoco_warp as mjwarp
from mujoco_warp._src import types as mjwarp_types
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
_s = mjwarp.Statistic(
    **{f.name: None for f in dataclasses.fields(mjwarp.Statistic) if f.init}
)
_c = mjwarp.Contact(
    **{f.name: None for f in dataclasses.fields(mjwarp.Contact) if f.init}
)
_e = mjwarp.Constraint(
    **{f.name: None for f in dataclasses.fields(mjwarp.Constraint) if f.init}
)


def _collision(
    # Model
    ngeom: int,
    opt__disableflags: int,
    opt__gjk_iterations: int,
    opt__epa_iterations: int,
    opt__epa_exact_neg_distance: bool,
    opt__depth_extension: float,
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    nxn_geom_pair: wp.array(dtype=wp.vec2i),
    nxn_pairid: wp.array(dtype=int),
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=mjwarp_types.vec5),
    # Data
    nconmax: int,
    ncon: wp.array(dtype=int),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    sap_cumulative_sum: wp.array(dtype=int),
    sap_segment_index: wp.array(dtype=int),
    ncollision: wp.array(dtype=int),
    contact__dist: wp.array(dtype=float),
    contact__pos: wp.array(dtype=wp.vec3),
    contact__frame: wp.array(dtype=wp.mat33),
    contact__includemargin: wp.array(dtype=float),
    contact__friction: wp.array(dtype=mjwarp_types.vec5),
    contact__solref: wp.array(dtype=wp.vec2),
    contact__solreffriction: wp.array(dtype=wp.vec2),
    contact__solimp: wp.array(dtype=mjwarp_types.vec5),
    contact__dim: wp.array(dtype=int),
    contact__geom: wp.array(dtype=wp.vec2i),
    contact__worldid: wp.array(dtype=int),
    sap_projection_lower: wp.array2d(dtype=float),
    sap_projection_upper: wp.array2d(dtype=float),
    sap_sort_index: wp.array2d(dtype=int),
    sap_range: wp.array2d(dtype=int),
    collision_pair: wp.array(dtype=wp.vec2i),
    collision_pairid: wp.array(dtype=int),
    collision_worldid: wp.array(dtype=int),
):
  _m.stat = _s
  _m.opt = _o
  _d.efc = _e
  _d.contact = _c
  _m.geom_condim = geom_condim
  _m.geom_dataid = geom_dataid
  _m.geom_friction = geom_friction
  _m.geom_gap = geom_gap
  _m.geom_margin = geom_margin
  _m.geom_priority = geom_priority
  _m.geom_rbound = geom_rbound
  _m.geom_size = geom_size
  _m.geom_solimp = geom_solimp
  _m.geom_solmix = geom_solmix
  _m.geom_solref = geom_solref
  _m.geom_type = geom_type
  _m.mesh_vert = mesh_vert
  _m.mesh_vertadr = mesh_vertadr
  _m.mesh_vertnum = mesh_vertnum
  _m.ngeom = ngeom
  _m.nxn_geom_pair = nxn_geom_pair
  _m.nxn_pairid = nxn_pairid
  _m.opt.depth_extension = opt__depth_extension
  _m.opt.disableflags = opt__disableflags
  _m.opt.epa_exact_neg_distance = opt__epa_exact_neg_distance
  _m.opt.epa_iterations = opt__epa_iterations
  _m.opt.gjk_iterations = opt__gjk_iterations
  _m.pair_dim = pair_dim
  _m.pair_friction = pair_friction
  _m.pair_gap = pair_gap
  _m.pair_margin = pair_margin
  _m.pair_solimp = pair_solimp
  _m.pair_solref = pair_solref
  _m.pair_solreffriction = pair_solreffriction
  _d.collision_pair = collision_pair
  _d.collision_pairid = collision_pairid
  _d.collision_worldid = collision_worldid
  _d.contact.dim = contact__dim
  _d.contact.dist = contact__dist
  _d.contact.frame = contact__frame
  _d.contact.friction = contact__friction
  _d.contact.geom = contact__geom
  _d.contact.includemargin = contact__includemargin
  _d.contact.pos = contact__pos
  _d.contact.solimp = contact__solimp
  _d.contact.solref = contact__solref
  _d.contact.solreffriction = contact__solreffriction
  _d.contact.worldid = contact__worldid
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.ncollision = ncollision
  _d.ncon = ncon
  _d.nconmax = nconmax
  _d.sap_cumulative_sum = sap_cumulative_sum
  _d.sap_projection_lower = sap_projection_lower
  _d.sap_projection_upper = sap_projection_upper
  _d.sap_range = sap_range
  _d.sap_segment_index = sap_segment_index
  _d.sap_sort_index = sap_sort_index
  _d.nworld = _d.qpos.shape[0]
  mjwarp.collision(_m, _d)


def _collision_shim(
    ngeom: int,
    opt__disableflags: int,
    opt__gjk_iterations: int,
    opt__epa_iterations: int,
    opt__epa_exact_neg_distance: bool,
    opt__depth_extension: float,
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array(dtype=float),
    geom_solref: wp.array(dtype=wp.vec2),
    geom_solimp: wp.array(dtype=mjwarp_types.vec5),
    geom_size: wp.array(dtype=wp.vec3),
    geom_rbound: wp.array(dtype=float),
    geom_friction: wp.array(dtype=wp.vec3),
    geom_margin: wp.array(dtype=float),
    geom_gap: wp.array(dtype=float),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    nxn_geom_pair: wp.array(dtype=wp.vec2i),
    nxn_pairid: wp.array(dtype=int),
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array(dtype=wp.vec2),
    pair_solreffriction: wp.array(dtype=wp.vec2),
    pair_solimp: wp.array(dtype=mjwarp_types.vec5),
    pair_margin: wp.array(dtype=float),
    pair_gap: wp.array(dtype=float),
    pair_friction: wp.array(dtype=mjwarp_types.vec5),
    nconmax: int,
    ncon: wp.array(dtype=int),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    sap_cumulative_sum: wp.array(dtype=int),
    sap_segment_index: wp.array(dtype=int),
    ncollision: wp.array(dtype=int),
    contact__dist: wp.array(dtype=float),
    contact__pos: wp.array(dtype=wp.vec3),
    contact__frame: wp.array(dtype=wp.mat33),
    contact__includemargin: wp.array(dtype=float),
    contact__friction: wp.array(dtype=mjwarp_types.vec5),
    contact__solref: wp.array(dtype=wp.vec2),
    contact__solreffriction: wp.array(dtype=wp.vec2),
    contact__solimp: wp.array(dtype=mjwarp_types.vec5),
    contact__dim: wp.array(dtype=int),
    contact__geom: wp.array(dtype=wp.vec2i),
    contact__worldid: wp.array(dtype=int),
    sap_projection_lower: wp.array2d(dtype=float),
    sap_projection_upper: wp.array(dtype=float),
    sap_sort_index: wp.array2d(dtype=int),
    sap_range: wp.array(dtype=int),
    collision_pair: wp.array(dtype=wp.vec2i),
    collision_pairid: wp.array(dtype=int),
    collision_worldid: wp.array(dtype=int),
):
  args = (
      ngeom,
      opt__disableflags,
      opt__gjk_iterations,
      opt__epa_iterations,
      opt__epa_exact_neg_distance,
      opt__depth_extension,
      geom_type,
      geom_condim,
      geom_dataid,
      geom_priority,
      geom_solmix,
      geom_solref,
      geom_solimp,
      geom_size,
      geom_rbound,
      geom_friction,
      geom_margin,
      geom_gap,
      mesh_vertadr,
      mesh_vertnum,
      mesh_vert,
      nxn_geom_pair,
      nxn_pairid,
      pair_dim,
      pair_solref,
      pair_solreffriction,
      pair_solimp,
      pair_margin,
      pair_gap,
      pair_friction,
      nconmax,
      ncon,
      geom_xpos,
      geom_xmat,
      sap_cumulative_sum,
      sap_segment_index,
      ncollision,
      contact__dist,
      contact__pos,
      contact__frame,
      contact__includemargin,
      contact__friction,
      contact__solref,
      contact__solreffriction,
      contact__solimp,
      contact__dim,
      contact__geom,
      contact__worldid,
      sap_projection_lower,
      sap_projection_upper,
      sap_sort_index,
      sap_range,
      collision_pair,
      collision_pairid,
      collision_worldid,
  )
  names = (
      "ngeom",
      "opt__disableflags",
      "opt__gjk_iterations",
      "opt__epa_iterations",
      "opt__epa_exact_neg_distance",
      "opt__depth_extension",
      "geom_type",
      "geom_condim",
      "geom_dataid",
      "geom_priority",
      "geom_solmix",
      "geom_solref",
      "geom_solimp",
      "geom_size",
      "geom_rbound",
      "geom_friction",
      "geom_margin",
      "geom_gap",
      "mesh_vertadr",
      "mesh_vertnum",
      "mesh_vert",
      "nxn_geom_pair",
      "nxn_pairid",
      "pair_dim",
      "pair_solref",
      "pair_solreffriction",
      "pair_solimp",
      "pair_margin",
      "pair_gap",
      "pair_friction",
      "nconmax",
      "ncon",
      "geom_xpos",
      "geom_xmat",
      "sap_cumulative_sum",
      "sap_segment_index",
      "ncollision",
      "contact__dist",
      "contact__pos",
      "contact__frame",
      "contact__includemargin",
      "contact__friction",
      "contact__solref",
      "contact__solreffriction",
      "contact__solimp",
      "contact__dim",
      "contact__geom",
      "contact__worldid",
      "sap_projection_lower",
      "sap_projection_upper",
      "sap_sort_index",
      "sap_range",
      "collision_pair",
      "collision_pairid",
      "collision_worldid",
  )
  args = ffi_helper.format_args_for_warp(*args, names=names, kernel=_collision)
  _collision(*args)


def collision(m: types.Model, d: types.Data):
  output_dims = {
      "contact__dist": d._impl.contact__dist.shape,
      "contact__pos": d._impl.contact__pos.shape,
      "contact__frame": d._impl.contact__frame.shape,
      "contact__includemargin": d._impl.contact__includemargin.shape,
      "contact__friction": d._impl.contact__friction.shape,
      "contact__solref": d._impl.contact__solref.shape,
      "contact__solreffriction": d._impl.contact__solreffriction.shape,
      "contact__solimp": d._impl.contact__solimp.shape,
      "contact__dim": d._impl.contact__dim.shape,
      "contact__geom": d._impl.contact__geom.shape,
      "contact__worldid": d._impl.contact__worldid.shape,
      "sap_projection_lower": d._impl.sap_projection_lower.shape,
      "sap_projection_upper": d._impl.sap_projection_upper.shape,
      "sap_sort_index": d._impl.sap_sort_index.shape,
      "sap_range": d._impl.sap_range.shape,
      "collision_pair": d._impl.collision_pair.shape,
      "collision_pairid": d._impl.collision_pairid.shape,
      "collision_worldid": d._impl.collision_worldid.shape,
  }

  jf = ffi_helper.jax_callable_variadic_tuple(
      _collision_shim,
      num_outputs=18,
      output_dims=output_dims,
      vmap_method="expand_dims",
      graph_compatible=True,
  )
  out = jf(
      m.ngeom,
      m.opt.disableflags,
      m.opt.gjk_iterations,
      m.opt.epa_iterations,
      m.opt.epa_exact_neg_distance,
      m.opt.depth_extension,
      m.geom_type,
      m.geom_condim,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_rbound,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_vert,
      m._impl.nxn_geom_pair,
      m._impl.nxn_pairid,
      m.pair_dim,
      m.pair_solref,
      m.pair_solreffriction,
      m.pair_solimp,
      m.pair_margin,
      m.pair_gap,
      m.pair_friction,
      d._impl.nconmax,
      d._impl.ncon,
      d.geom_xpos,
      d.geom_xmat,
      d._impl.sap_cumulative_sum,
      d._impl.sap_segment_index,
      d._impl.ncollision,
  )
  d = d.tree_replace({
      "_impl.contact__dist": out[0],
      "_impl.contact__pos": out[1],
      "_impl.contact__frame": out[2],
      "_impl.contact__includemargin": out[3],
      "_impl.contact__friction": out[4],
      "_impl.contact__solref": out[5],
      "_impl.contact__solreffriction": out[6],
      "_impl.contact__solimp": out[7],
      "_impl.contact__dim": out[8],
      "_impl.contact__geom": out[9],
      "_impl.contact__worldid": out[10],
      "_impl.sap_projection_lower": out[11],
      "_impl.sap_projection_upper": out[12],
      "_impl.sap_sort_index": out[13],
      "_impl.sap_range": out[14],
      "_impl.collision_pair": out[15],
      "_impl.collision_pairid": out[16],
      "_impl.collision_worldid": out[17],
  })
  return d
