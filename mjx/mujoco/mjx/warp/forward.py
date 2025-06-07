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
import mujoco_warp as mjwarp
from mujoco_warp._src import types as mjwarp_types
import warp as wp

# TODO(btaba): create _m/_d inside the function scope, or use a mutex?
# which is faster?
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


@ffi.format_args_for_warp
def _forward_shim(
    # Model
    nworld: int,
    nv: int,
    nu: int,
    na: int,
    nbody: int,
    njnt: int,
    ngeom: int,
    nsite: int,
    ncam: int,
    nlight: int,
    nflexelem: int,
    neq: int,
    nmocap: int,
    ngravcomp: int,
    nM: int,
    ntendon: int,
    nlsp: int,
    opt__timestep: float,
    opt__impratio: float,
    opt__tolerance: float,
    opt__ls_tolerance: float,
    opt__gravity: wp.array(dtype=wp.vec3),
    opt__cone: int,
    opt__solver: int,
    opt__iterations: int,
    opt__ls_iterations: int,
    opt__disableflags: int,
    opt__is_sparse: bool,
    opt__ls_parallel: bool,
    opt__wind: wp.array(dtype=wp.vec3),
    opt__has_wind: bool,
    opt__density: float,
    opt__viscosity: float,
    stat__meaninertia: float,
    qpos0: wp.array2d(dtype=float),
    qpos_spring: wp.array2d(dtype=float),
    qM_fullm_i: wp.array(dtype=int),
    qM_fullm_j: wp.array(dtype=int),
    qM_mulm_i: wp.array(dtype=int),
    qM_mulm_j: wp.array(dtype=int),
    qM_madr_ij: wp.array(dtype=int),
    qLD_updates: tuple[wp.array(dtype=wp.vec3i), ...],
    M_rownnz: wp.array(dtype=int),
    M_rowadr: wp.array(dtype=int),
    mapM2M: wp.array(dtype=int),
    qM_tiles: tuple[mjwarp_types.TileSet, ...],
    body_tree: tuple[wp.array(dtype=int), ...],
    body_parentid: wp.array(dtype=int),
    body_rootid: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_dofnum: wp.array(dtype=int),
    body_dofadr: wp.array(dtype=int),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    body_mass: wp.array2d(dtype=float),
    body_subtreemass: wp.array2d(dtype=float),
    subtree_mass: wp.array2d(dtype=float),
    body_inertia: wp.array2d(dtype=wp.vec3),
    body_invweight0: wp.array3d(dtype=float),
    body_gravcomp: wp.array2d(dtype=float),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_dofadr: wp.array(dtype=int),
    jnt_bodyid: wp.array(dtype=int),
    jnt_actfrclimited: wp.array(dtype=bool),
    jnt_solref: wp.array2d(dtype=wp.vec2),
    jnt_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    jnt_stiffness: wp.array2d(dtype=float),
    jnt_range: wp.array3d(dtype=float),
    jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
    jnt_margin: wp.array2d(dtype=float),
    jnt_limited_slide_hinge_adr: wp.array(dtype=int),
    jnt_limited_ball_adr: wp.array(dtype=int),
    jnt_actgravcomp: wp.array(dtype=int),
    dof_bodyid: wp.array(dtype=int),
    dof_jntid: wp.array(dtype=int),
    dof_parentid: wp.array(dtype=int),
    dof_Madr: wp.array(dtype=int),
    dof_armature: wp.array2d(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    dof_invweight0: wp.array2d(dtype=float),
    dof_frictionloss: wp.array2d(dtype=float),
    dof_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    dof_solref: wp.array2d(dtype=wp.vec2),
    dof_tri_row: wp.array(dtype=int),
    dof_tri_col: wp.array(dtype=int),
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_bodyid: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    site_type: wp.array(dtype=int),
    site_bodyid: wp.array(dtype=int),
    site_size: wp.array(dtype=wp.vec3),
    site_pos: wp.array2d(dtype=wp.vec3),
    site_quat: wp.array2d(dtype=wp.quat),
    cam_mode: wp.array(dtype=int),
    cam_bodyid: wp.array(dtype=int),
    cam_targetbodyid: wp.array(dtype=int),
    cam_pos: wp.array2d(dtype=wp.vec3),
    cam_quat: wp.array2d(dtype=wp.quat),
    cam_poscom0: wp.array2d(dtype=wp.vec3),
    cam_pos0: wp.array2d(dtype=wp.vec3),
    cam_mat0: wp.array2d(dtype=wp.mat33),
    cam_fovy: wp.array(dtype=float),
    cam_resolution: wp.array(dtype=wp.vec2i),
    cam_sensorsize: wp.array(dtype=wp.vec2),
    cam_intrinsic: wp.array(dtype=wp.vec4),
    light_mode: wp.array(dtype=int),
    light_bodyid: wp.array(dtype=int),
    light_targetbodyid: wp.array(dtype=int),
    light_pos: wp.array2d(dtype=wp.vec3),
    light_dir: wp.array2d(dtype=wp.vec3),
    light_poscom0: wp.array2d(dtype=wp.vec3),
    light_pos0: wp.array2d(dtype=wp.vec3),
    light_dir0: wp.array2d(dtype=wp.vec3),
    flex_dim: wp.array(dtype=int),
    flex_vertadr: wp.array(dtype=int),
    flex_edgeadr: wp.array(dtype=int),
    flex_elemedgeadr: wp.array(dtype=int),
    flex_vertbodyid: wp.array(dtype=int),
    flex_elem: wp.array(dtype=int),
    flex_elemedge: wp.array(dtype=int),
    flexedge_length0: wp.array(dtype=float),
    flex_stiffness: wp.array(dtype=float),
    flex_damping: wp.array(dtype=float),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    eq_obj1id: wp.array(dtype=int),
    eq_obj2id: wp.array(dtype=int),
    eq_objtype: wp.array(dtype=int),
    eq_solref: wp.array2d(dtype=wp.vec2),
    eq_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    eq_data: wp.array2d(dtype=mjwarp_types.vec11),
    eq_connect_adr: wp.array(dtype=int),
    eq_wld_adr: wp.array(dtype=int),
    eq_jnt_adr: wp.array(dtype=int),
    eq_ten_adr: wp.array(dtype=int),
    actuator_moment_tiles_nv: tuple[mjwarp_types.TileSet, ...],
    actuator_moment_tiles_nu: tuple[mjwarp_types.TileSet, ...],
    actuator_trntype: wp.array(dtype=int),
    actuator_dyntype: wp.array(dtype=int),
    actuator_gaintype: wp.array(dtype=int),
    actuator_biastype: wp.array(dtype=int),
    actuator_trnid: wp.array(dtype=wp.vec2i),
    actuator_actadr: wp.array(dtype=int),
    actuator_actnum: wp.array(dtype=int),
    actuator_ctrllimited: wp.array(dtype=bool),
    actuator_forcelimited: wp.array(dtype=bool),
    actuator_dynprm: wp.array2d(dtype=mjwarp_types.vec10f),
    actuator_gainprm: wp.array2d(dtype=mjwarp_types.vec10f),
    actuator_biasprm: wp.array2d(dtype=mjwarp_types.vec10f),
    actuator_ctrlrange: wp.array2d(dtype=wp.vec2),
    actuator_forcerange: wp.array2d(dtype=wp.vec2),
    actuator_gear: wp.array2d(dtype=wp.spatial_vector),
    nxn_geom_pair: wp.array(dtype=wp.vec2i),
    nxn_pairid: wp.array(dtype=int),
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=mjwarp_types.vec5),
    condim_max: int,
    tendon_adr: wp.array(dtype=int),
    tendon_num: wp.array(dtype=int),
    tendon_limited_adr: wp.array(dtype=int),
    tendon_solref_lim: wp.array2d(dtype=wp.vec2),
    tendon_solimp_lim: wp.array2d(dtype=mjwarp_types.vec5),
    tendon_solref_fri: wp.array2d(dtype=wp.vec2),
    tendon_solimp_fri: wp.array2d(dtype=mjwarp_types.vec5),
    tendon_range: wp.array2d(dtype=wp.vec2),
    tendon_margin: wp.array2d(dtype=float),
    tendon_stiffness: wp.array2d(dtype=float),
    tendon_damping: wp.array2d(dtype=float),
    tendon_frictionloss: wp.array2d(dtype=float),
    tendon_lengthspring: wp.array2d(dtype=wp.vec2),
    tendon_length0: wp.array2d(dtype=float),
    tendon_invweight0: wp.array2d(dtype=float),
    wrap_objid: wp.array(dtype=int),
    wrap_prm: wp.array(dtype=float),
    wrap_type: wp.array(dtype=int),
    tendon_jnt_adr: wp.array(dtype=int),
    tendon_site_pair_adr: wp.array(dtype=int),
    ten_wrapadr_site: wp.array(dtype=int),
    ten_wrapnum_site: wp.array(dtype=int),
    wrap_jnt_adr: wp.array(dtype=int),
    wrap_site_adr: wp.array(dtype=int),
    wrap_site_pair_adr: wp.array(dtype=int),
    sensor_type: wp.array(dtype=int),
    sensor_datatype: wp.array(dtype=int),
    sensor_objtype: wp.array(dtype=int),
    sensor_objid: wp.array(dtype=int),
    sensor_reftype: wp.array(dtype=int),
    sensor_refid: wp.array(dtype=int),
    sensor_adr: wp.array(dtype=int),
    sensor_cutoff: wp.array(dtype=float),
    sensor_pos_adr: wp.array(dtype=int),
    sensor_vel_adr: wp.array(dtype=int),
    sensor_acc_adr: wp.array(dtype=int),
    sensor_touch_adr: wp.array(dtype=int),
    sensor_subtree_vel: bool,
    sensor_rne_postconstraint: bool,
    mocap_bodyid: wp.array(dtype=int),
    # Data
    nconmax: int,
    njmax: int,
    time: wp.array(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    qacc_warmstart: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    qfrc_applied: wp.array2d(dtype=float),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
    eq_active: wp.array2d(dtype=bool),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
    flexvert_xpos: wp.array2d(dtype=wp.vec3),
    flexedge_length: wp.array2d(dtype=float),
    flexedge_velocity: wp.array2d(dtype=float),
    qLD: wp.array3d(dtype=float),
    qLDiagInv: wp.array2d(dtype=float),
    qfrc_fluid: wp.array2d(dtype=float),
    qacc_smooth: wp.array2d(dtype=float),
    efc__mv: wp.array2d(dtype=float),
    ncon: wp.array(dtype=int),
    ne: wp.array(dtype=int),
    ne_connect: wp.array(dtype=int),
    ne_weld: wp.array(dtype=int),
    ne_jnt: wp.array(dtype=int),
    ne_ten: wp.array(dtype=int),
    nf: wp.array(dtype=int),
    nl: wp.array(dtype=int),
    nefc: wp.array(dtype=int),
    fluid_applied: wp.array2d(dtype=wp.spatial_vector),
    qacc: wp.array2d(dtype=float),
    act_dot: wp.array2d(dtype=float),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    xmat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3),
    ximat: wp.array2d(dtype=wp.mat33),
    xanchor: wp.array2d(dtype=wp.vec3),
    xaxis: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    cinert: wp.array2d(dtype=mjwarp_types.vec10),
    actuator_length: wp.array2d(dtype=float),
    actuator_moment: wp.array3d(dtype=float),
    crb: wp.array2d(dtype=mjwarp_types.vec10),
    qM: wp.array3d(dtype=float),
    ten_velocity: wp.array2d(dtype=float),
    actuator_velocity: wp.array2d(dtype=float),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    cdof_dot: wp.array2d(dtype=wp.spatial_vector),
    qfrc_bias: wp.array2d(dtype=float),
    qfrc_spring: wp.array2d(dtype=float),
    qfrc_damper: wp.array2d(dtype=float),
    qfrc_gravcomp: wp.array2d(dtype=float),
    qfrc_passive: wp.array2d(dtype=float),
    subtree_linvel: wp.array2d(dtype=wp.vec3),
    subtree_angmom: wp.array2d(dtype=wp.vec3),
    subtree_bodyvel: wp.array2d(dtype=wp.spatial_vector),
    actuator_force: wp.array2d(dtype=float),
    qfrc_actuator: wp.array2d(dtype=float),
    qfrc_smooth: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
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
    contact__efc_address: wp.array2d(dtype=int),
    contact__worldid: wp.array(dtype=int),
    efc__worldid: wp.array(dtype=int),
    efc__id: wp.array(dtype=int),
    efc__J: wp.array2d(dtype=float),
    efc__pos: wp.array(dtype=float),
    efc__margin: wp.array(dtype=float),
    efc__D: wp.array(dtype=float),
    efc__aref: wp.array(dtype=float),
    efc__frictionloss: wp.array(dtype=float),
    efc__force: wp.array(dtype=float),
    efc__Jaref: wp.array(dtype=float),
    efc__Ma: wp.array2d(dtype=float),
    efc__grad: wp.array2d(dtype=float),
    efc__grad_dot: wp.array(dtype=float),
    efc__Mgrad: wp.array2d(dtype=float),
    efc__search: wp.array2d(dtype=float),
    efc__search_dot: wp.array(dtype=float),
    efc__gauss: wp.array(dtype=float),
    efc__cost: wp.array(dtype=float),
    efc__prev_cost: wp.array(dtype=float),
    efc__solver_niter: wp.array(dtype=int),
    efc__active: wp.array(dtype=bool),
    efc__gtol: wp.array(dtype=float),
    efc__jv: wp.array(dtype=float),
    efc__quad: wp.array(dtype=wp.vec3),
    efc__quad_gauss: wp.array(dtype=wp.vec3),
    efc__h: wp.array3d(dtype=float),
    efc__alpha: wp.array(dtype=float),
    efc__prev_grad: wp.array2d(dtype=float),
    efc__prev_Mgrad: wp.array2d(dtype=float),
    efc__beta: wp.array(dtype=float),
    efc__beta_num: wp.array(dtype=float),
    efc__beta_den: wp.array(dtype=float),
    efc__done: wp.array(dtype=bool),
    efc__ls_done: wp.array(dtype=bool),
    efc__p0: wp.array(dtype=wp.vec3),
    efc__lo: wp.array(dtype=wp.vec3),
    efc__lo_alpha: wp.array(dtype=float),
    efc__hi: wp.array(dtype=wp.vec3),
    efc__hi_alpha: wp.array(dtype=float),
    efc__lo_next: wp.array(dtype=wp.vec3),
    efc__lo_next_alpha: wp.array(dtype=float),
    efc__hi_next: wp.array(dtype=wp.vec3),
    efc__hi_next_alpha: wp.array(dtype=float),
    efc__mid: wp.array(dtype=wp.vec3),
    efc__mid_alpha: wp.array(dtype=float),
    efc__cost_candidate: wp.array2d(dtype=float),
    efc__quad_total_candidate: wp.array2d(dtype=wp.vec3),
    efc__u: wp.array(dtype=mjwarp_types.vec6),
    efc__uu: wp.array(dtype=float),
    efc__uv: wp.array(dtype=float),
    efc__vv: wp.array(dtype=float),
    efc__condim: wp.array(dtype=int),
    collision_pair: wp.array(dtype=wp.vec2i),
    collision_pairid: wp.array(dtype=int),
    collision_worldid: wp.array(dtype=int),
    ncollision: wp.array(dtype=int),
    cacc: wp.array2d(dtype=wp.spatial_vector),
    cfrc_int: wp.array2d(dtype=wp.spatial_vector),
    cfrc_ext: wp.array2d(dtype=wp.spatial_vector),
    ten_length: wp.array2d(dtype=float),
    ten_J: wp.array3d(dtype=float),
    ten_wrapadr: wp.array2d(dtype=int),
    ten_wrapnum: wp.array2d(dtype=int),
    wrap_obj: wp.array2d(dtype=wp.vec2i),
    wrap_xpos: wp.array2d(dtype=wp.spatial_vector),
    sensordata: wp.array2d(dtype=float),
):
  _m.stat = _s
  _m.opt = _o
  _d.efc = _e
  _d.contact = _c
  _m.M_rowadr = M_rowadr
  _m.M_rownnz = M_rownnz
  _m.actuator_actadr = actuator_actadr
  _m.actuator_actnum = actuator_actnum
  _m.actuator_biasprm = actuator_biasprm
  _m.actuator_biastype = actuator_biastype
  _m.actuator_ctrllimited = actuator_ctrllimited
  _m.actuator_ctrlrange = actuator_ctrlrange
  _m.actuator_dynprm = actuator_dynprm
  _m.actuator_dyntype = actuator_dyntype
  _m.actuator_forcelimited = actuator_forcelimited
  _m.actuator_forcerange = actuator_forcerange
  _m.actuator_gainprm = actuator_gainprm
  _m.actuator_gaintype = actuator_gaintype
  _m.actuator_gear = actuator_gear
  _m.actuator_moment_tiles_nu = actuator_moment_tiles_nu
  _m.actuator_moment_tiles_nv = actuator_moment_tiles_nv
  _m.actuator_trnid = actuator_trnid
  _m.actuator_trntype = actuator_trntype
  _m.body_dofadr = body_dofadr
  _m.body_dofnum = body_dofnum
  _m.body_gravcomp = body_gravcomp
  _m.body_inertia = body_inertia
  _m.body_invweight0 = body_invweight0
  _m.body_ipos = body_ipos
  _m.body_iquat = body_iquat
  _m.body_jntadr = body_jntadr
  _m.body_jntnum = body_jntnum
  _m.body_mass = body_mass
  _m.body_parentid = body_parentid
  _m.body_pos = body_pos
  _m.body_quat = body_quat
  _m.body_rootid = body_rootid
  _m.body_subtreemass = body_subtreemass
  _m.body_tree = body_tree
  _m.cam_bodyid = cam_bodyid
  _m.cam_fovy = cam_fovy
  _m.cam_intrinsic = cam_intrinsic
  _m.cam_mat0 = cam_mat0
  _m.cam_mode = cam_mode
  _m.cam_pos = cam_pos
  _m.cam_pos0 = cam_pos0
  _m.cam_poscom0 = cam_poscom0
  _m.cam_quat = cam_quat
  _m.cam_resolution = cam_resolution
  _m.cam_sensorsize = cam_sensorsize
  _m.cam_targetbodyid = cam_targetbodyid
  _m.condim_max = condim_max
  _m.dof_Madr = dof_Madr
  _m.dof_armature = dof_armature
  _m.dof_bodyid = dof_bodyid
  _m.dof_damping = dof_damping
  _m.dof_frictionloss = dof_frictionloss
  _m.dof_invweight0 = dof_invweight0
  _m.dof_jntid = dof_jntid
  _m.dof_parentid = dof_parentid
  _m.dof_solimp = dof_solimp
  _m.dof_solref = dof_solref
  _m.dof_tri_col = dof_tri_col
  _m.dof_tri_row = dof_tri_row
  _m.eq_connect_adr = eq_connect_adr
  _m.eq_data = eq_data
  _m.eq_jnt_adr = eq_jnt_adr
  _m.eq_obj1id = eq_obj1id
  _m.eq_obj2id = eq_obj2id
  _m.eq_objtype = eq_objtype
  _m.eq_solimp = eq_solimp
  _m.eq_solref = eq_solref
  _m.eq_ten_adr = eq_ten_adr
  _m.eq_wld_adr = eq_wld_adr
  _m.flex_damping = flex_damping
  _m.flex_dim = flex_dim
  _m.flex_edgeadr = flex_edgeadr
  _m.flex_elem = flex_elem
  _m.flex_elemedge = flex_elemedge
  _m.flex_elemedgeadr = flex_elemedgeadr
  _m.flex_stiffness = flex_stiffness
  _m.flex_vertadr = flex_vertadr
  _m.flex_vertbodyid = flex_vertbodyid
  _m.flexedge_length0 = flexedge_length0
  _m.geom_bodyid = geom_bodyid
  _m.geom_condim = geom_condim
  _m.geom_dataid = geom_dataid
  _m.geom_friction = geom_friction
  _m.geom_gap = geom_gap
  _m.geom_margin = geom_margin
  _m.geom_pos = geom_pos
  _m.geom_priority = geom_priority
  _m.geom_quat = geom_quat
  _m.geom_rbound = geom_rbound
  _m.geom_size = geom_size
  _m.geom_solimp = geom_solimp
  _m.geom_solmix = geom_solmix
  _m.geom_solref = geom_solref
  _m.geom_type = geom_type
  _m.jnt_actfrclimited = jnt_actfrclimited
  _m.jnt_actfrcrange = jnt_actfrcrange
  _m.jnt_actgravcomp = jnt_actgravcomp
  _m.jnt_axis = jnt_axis
  _m.jnt_bodyid = jnt_bodyid
  _m.jnt_dofadr = jnt_dofadr
  _m.jnt_limited_ball_adr = jnt_limited_ball_adr
  _m.jnt_limited_slide_hinge_adr = jnt_limited_slide_hinge_adr
  _m.jnt_margin = jnt_margin
  _m.jnt_pos = jnt_pos
  _m.jnt_qposadr = jnt_qposadr
  _m.jnt_range = jnt_range
  _m.jnt_solimp = jnt_solimp
  _m.jnt_solref = jnt_solref
  _m.jnt_stiffness = jnt_stiffness
  _m.jnt_type = jnt_type
  _m.light_bodyid = light_bodyid
  _m.light_dir = light_dir
  _m.light_dir0 = light_dir0
  _m.light_mode = light_mode
  _m.light_pos = light_pos
  _m.light_pos0 = light_pos0
  _m.light_poscom0 = light_poscom0
  _m.light_targetbodyid = light_targetbodyid
  _m.mapM2M = mapM2M
  _m.mesh_vert = mesh_vert
  _m.mesh_vertadr = mesh_vertadr
  _m.mesh_vertnum = mesh_vertnum
  _m.mocap_bodyid = mocap_bodyid
  _m.nM = nM
  _m.na = na
  _m.nbody = nbody
  _m.ncam = ncam
  _m.neq = neq
  _m.nflexelem = nflexelem
  _m.ngeom = ngeom
  _m.ngravcomp = ngravcomp
  _m.njnt = njnt
  _m.nlight = nlight
  _m.nlsp = nlsp
  _m.nmocap = nmocap
  _m.nsite = nsite
  _m.ntendon = ntendon
  _m.nu = nu
  _m.nv = nv
  _m.nxn_geom_pair = nxn_geom_pair
  _m.nxn_pairid = nxn_pairid
  _m.opt.cone = opt__cone
  _m.opt.density = opt__density
  _m.opt.disableflags = opt__disableflags
  _m.opt.gravity = opt__gravity
  _m.opt.has_wind = opt__has_wind
  _m.opt.impratio = opt__impratio
  _m.opt.is_sparse = opt__is_sparse
  _m.opt.iterations = opt__iterations
  _m.opt.ls_iterations = opt__ls_iterations
  _m.opt.ls_parallel = opt__ls_parallel
  _m.opt.ls_tolerance = opt__ls_tolerance
  _m.opt.solver = opt__solver
  _m.opt.timestep = opt__timestep
  _m.opt.tolerance = opt__tolerance
  _m.opt.viscosity = opt__viscosity
  _m.opt.wind = opt__wind
  _m.pair_dim = pair_dim
  _m.pair_friction = pair_friction
  _m.pair_gap = pair_gap
  _m.pair_margin = pair_margin
  _m.pair_solimp = pair_solimp
  _m.pair_solref = pair_solref
  _m.pair_solreffriction = pair_solreffriction
  _m.qLD_updates = qLD_updates
  _m.qM_fullm_i = qM_fullm_i
  _m.qM_fullm_j = qM_fullm_j
  _m.qM_madr_ij = qM_madr_ij
  _m.qM_mulm_i = qM_mulm_i
  _m.qM_mulm_j = qM_mulm_j
  _m.qM_tiles = qM_tiles
  _m.qpos0 = qpos0
  _m.qpos_spring = qpos_spring
  _m.sensor_acc_adr = sensor_acc_adr
  _m.sensor_adr = sensor_adr
  _m.sensor_cutoff = sensor_cutoff
  _m.sensor_datatype = sensor_datatype
  _m.sensor_objid = sensor_objid
  _m.sensor_objtype = sensor_objtype
  _m.sensor_pos_adr = sensor_pos_adr
  _m.sensor_refid = sensor_refid
  _m.sensor_reftype = sensor_reftype
  _m.sensor_rne_postconstraint = sensor_rne_postconstraint
  _m.sensor_subtree_vel = sensor_subtree_vel
  _m.sensor_touch_adr = sensor_touch_adr
  _m.sensor_type = sensor_type
  _m.sensor_vel_adr = sensor_vel_adr
  _m.site_bodyid = site_bodyid
  _m.site_pos = site_pos
  _m.site_quat = site_quat
  _m.site_size = site_size
  _m.site_type = site_type
  _m.stat.meaninertia = stat__meaninertia
  _m.subtree_mass = subtree_mass
  _m.ten_wrapadr_site = ten_wrapadr_site
  _m.ten_wrapnum_site = ten_wrapnum_site
  _m.tendon_adr = tendon_adr
  _m.tendon_damping = tendon_damping
  _m.tendon_frictionloss = tendon_frictionloss
  _m.tendon_invweight0 = tendon_invweight0
  _m.tendon_jnt_adr = tendon_jnt_adr
  _m.tendon_length0 = tendon_length0
  _m.tendon_lengthspring = tendon_lengthspring
  _m.tendon_limited_adr = tendon_limited_adr
  _m.tendon_margin = tendon_margin
  _m.tendon_num = tendon_num
  _m.tendon_range = tendon_range
  _m.tendon_site_pair_adr = tendon_site_pair_adr
  _m.tendon_solimp_fri = tendon_solimp_fri
  _m.tendon_solimp_lim = tendon_solimp_lim
  _m.tendon_solref_fri = tendon_solref_fri
  _m.tendon_solref_lim = tendon_solref_lim
  _m.tendon_stiffness = tendon_stiffness
  _m.wrap_jnt_adr = wrap_jnt_adr
  _m.wrap_objid = wrap_objid
  _m.wrap_prm = wrap_prm
  _m.wrap_site_adr = wrap_site_adr
  _m.wrap_site_pair_adr = wrap_site_pair_adr
  _m.wrap_type = wrap_type
  _d.act = act
  _d.act_dot = act_dot
  _d.actuator_force = actuator_force
  _d.actuator_length = actuator_length
  _d.actuator_moment = actuator_moment
  _d.actuator_velocity = actuator_velocity
  _d.cacc = cacc
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.cdof = cdof
  _d.cdof_dot = cdof_dot
  _d.cfrc_ext = cfrc_ext
  _d.cfrc_int = cfrc_int
  _d.cinert = cinert
  _d.collision_pair = collision_pair
  _d.collision_pairid = collision_pairid
  _d.collision_worldid = collision_worldid
  _d.contact.dim = contact__dim
  _d.contact.dist = contact__dist
  _d.contact.efc_address = contact__efc_address
  _d.contact.frame = contact__frame
  _d.contact.friction = contact__friction
  _d.contact.geom = contact__geom
  _d.contact.includemargin = contact__includemargin
  _d.contact.pos = contact__pos
  _d.contact.solimp = contact__solimp
  _d.contact.solref = contact__solref
  _d.contact.solreffriction = contact__solreffriction
  _d.contact.worldid = contact__worldid
  _d.crb = crb
  _d.ctrl = ctrl
  _d.cvel = cvel
  _d.efc.D = efc__D
  _d.efc.J = efc__J
  _d.efc.Jaref = efc__Jaref
  _d.efc.Ma = efc__Ma
  _d.efc.Mgrad = efc__Mgrad
  _d.efc.active = efc__active
  _d.efc.alpha = efc__alpha
  _d.efc.aref = efc__aref
  _d.efc.beta = efc__beta
  _d.efc.beta_den = efc__beta_den
  _d.efc.beta_num = efc__beta_num
  _d.efc.condim = efc__condim
  _d.efc.cost = efc__cost
  _d.efc.cost_candidate = efc__cost_candidate
  _d.efc.done = efc__done
  _d.efc.force = efc__force
  _d.efc.frictionloss = efc__frictionloss
  _d.efc.gauss = efc__gauss
  _d.efc.grad = efc__grad
  _d.efc.grad_dot = efc__grad_dot
  _d.efc.gtol = efc__gtol
  _d.efc.h = efc__h
  _d.efc.hi = efc__hi
  _d.efc.hi_alpha = efc__hi_alpha
  _d.efc.hi_next = efc__hi_next
  _d.efc.hi_next_alpha = efc__hi_next_alpha
  _d.efc.id = efc__id
  _d.efc.jv = efc__jv
  _d.efc.lo = efc__lo
  _d.efc.lo_alpha = efc__lo_alpha
  _d.efc.lo_next = efc__lo_next
  _d.efc.lo_next_alpha = efc__lo_next_alpha
  _d.efc.ls_done = efc__ls_done
  _d.efc.margin = efc__margin
  _d.efc.mid = efc__mid
  _d.efc.mid_alpha = efc__mid_alpha
  _d.efc.mv = efc__mv
  _d.efc.p0 = efc__p0
  _d.efc.pos = efc__pos
  _d.efc.prev_Mgrad = efc__prev_Mgrad
  _d.efc.prev_cost = efc__prev_cost
  _d.efc.prev_grad = efc__prev_grad
  _d.efc.quad = efc__quad
  _d.efc.quad_gauss = efc__quad_gauss
  _d.efc.quad_total_candidate = efc__quad_total_candidate
  _d.efc.search = efc__search
  _d.efc.search_dot = efc__search_dot
  _d.efc.solver_niter = efc__solver_niter
  _d.efc.u = efc__u
  _d.efc.uu = efc__uu
  _d.efc.uv = efc__uv
  _d.efc.vv = efc__vv
  _d.efc.worldid = efc__worldid
  _d.eq_active = eq_active
  _d.flexedge_length = flexedge_length
  _d.flexedge_velocity = flexedge_velocity
  _d.flexvert_xpos = flexvert_xpos
  _d.fluid_applied = fluid_applied
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  _d.mocap_pos = mocap_pos
  _d.mocap_quat = mocap_quat
  _d.ncollision = ncollision
  _d.ncon = ncon
  _d.nconmax = nconmax
  _d.ne = ne
  _d.ne_connect = ne_connect
  _d.ne_jnt = ne_jnt
  _d.ne_ten = ne_ten
  _d.ne_weld = ne_weld
  _d.nefc = nefc
  _d.nf = nf
  _d.njmax = njmax
  _d.nl = nl
  _d.qLD = qLD
  _d.qLDiagInv = qLDiagInv
  _d.qM = qM
  _d.qacc = qacc
  _d.qacc_smooth = qacc_smooth
  _d.qacc_warmstart = qacc_warmstart
  _d.qfrc_actuator = qfrc_actuator
  _d.qfrc_applied = qfrc_applied
  _d.qfrc_bias = qfrc_bias
  _d.qfrc_constraint = qfrc_constraint
  _d.qfrc_damper = qfrc_damper
  _d.qfrc_fluid = qfrc_fluid
  _d.qfrc_gravcomp = qfrc_gravcomp
  _d.qfrc_passive = qfrc_passive
  _d.qfrc_smooth = qfrc_smooth
  _d.qfrc_spring = qfrc_spring
  _d.qpos = qpos
  _d.qvel = qvel
  _d.sensordata = sensordata
  _d.site_xmat = site_xmat
  _d.site_xpos = site_xpos
  _d.subtree_angmom = subtree_angmom
  _d.subtree_bodyvel = subtree_bodyvel
  _d.subtree_com = subtree_com
  _d.subtree_linvel = subtree_linvel
  _d.ten_J = ten_J
  _d.ten_length = ten_length
  _d.ten_velocity = ten_velocity
  _d.ten_wrapadr = ten_wrapadr
  _d.ten_wrapnum = ten_wrapnum
  _d.time = time
  _d.wrap_obj = wrap_obj
  _d.wrap_xpos = wrap_xpos
  _d.xanchor = xanchor
  _d.xaxis = xaxis
  _d.xfrc_applied = xfrc_applied
  _d.ximat = ximat
  _d.xipos = xipos
  _d.xmat = xmat
  _d.xpos = xpos
  _d.xquat = xquat
  _d.nworld = nworld
  mjwarp.forward(_m, _d)


def _forward_jax_impl(m: types.Model, d: types.Data):
  output_dims = {
      'qacc': d.qacc.shape,
      'act_dot': d.act_dot.shape,
      'xpos': d.xpos.shape,
      'xquat': d.xquat.shape,
      'xmat': d.xmat.shape,
      'xipos': d.xipos.shape,
      'ximat': d.ximat.shape,
      'xanchor': d.xanchor.shape,
      'xaxis': d.xaxis.shape,
      'geom_xpos': d.geom_xpos.shape,
      'geom_xmat': d.geom_xmat.shape,
      'site_xpos': d.site_xpos.shape,
      'site_xmat': d.site_xmat.shape,
      'cam_xpos': d.cam_xpos.shape,
      'cam_xmat': d.cam_xmat.shape,
      'subtree_com': d.subtree_com.shape,
      'cvel': d.cvel.shape,
      'qfrc_bias': d.qfrc_bias.shape,
      'qfrc_gravcomp': d.qfrc_gravcomp.shape,
      'qfrc_passive': d.qfrc_passive.shape,
      'actuator_force': d.actuator_force.shape,
      'qfrc_actuator': d.qfrc_actuator.shape,
      'qfrc_smooth': d.qfrc_smooth.shape,
      'qfrc_constraint': d.qfrc_constraint.shape,
      'sensordata': d.sensordata.shape,
  }

  jf = ffi.jax_callable_variadic_tuple(
      _forward_shim,
      num_outputs=25,
      output_dims=output_dims,
      vmap_method=None,
      graph_compatible=True,
      in_out_argnames={
          'qacc',
          'act_dot',
          'xpos',
          'xquat',
          'xmat',
          'xipos',
          'ximat',
          'xanchor',
          'xaxis',
          'geom_xpos',
          'geom_xmat',
          'site_xpos',
          'site_xmat',
          'cam_xpos',
          'cam_xmat',
          'subtree_com',
          'cvel',
          'qfrc_bias',
          'qfrc_gravcomp',
          'qfrc_passive',
          'actuator_force',
          'qfrc_actuator',
          'qfrc_smooth',
          'qfrc_constraint',
          'sensordata',
      },
  )
  out = jf(
      d.qpos.shape[0],
      m.nv,
      m.nu,
      m.na,
      m.nbody,
      m.njnt,
      m.ngeom,
      m.nsite,
      m.ncam,
      m.nlight,
      m._impl.nflexelem,
      m.neq,
      m.nmocap,
      m.ngravcomp,
      m.nM,
      m.ntendon,
      m._impl.nlsp,
      m.opt.timestep,
      m.opt.impratio,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.gravity,
      m.opt.cone,
      m.opt.solver,
      m.opt.iterations,
      m.opt.ls_iterations,
      m.opt.disableflags,
      m.opt.is_sparse,
      m.opt.ls_parallel,
      m.opt.wind,
      m.opt.has_wind,
      m.opt.density,
      m.opt.viscosity,
      m.stat.meaninertia,
      m.qpos0,
      m.qpos_spring,
      m._impl.qM_fullm_i,
      m._impl.qM_fullm_j,
      m._impl.qM_mulm_i,
      m._impl.qM_mulm_j,
      m._impl.qM_madr_ij,
      m._impl.qLD_updates,
      m._impl.M_rownnz,
      m._impl.M_rowadr,
      m._impl.mapM2M,
      m._impl.qM_tiles,
      m._impl.body_tree,
      m.body_parentid,
      m.body_rootid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_dofnum,
      m.body_dofadr,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.body_mass,
      m.body_subtreemass,
      m._impl.subtree_mass,
      m.body_inertia,
      m.body_invweight0,
      m.body_gravcomp,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.jnt_actfrclimited,
      m.jnt_solref,
      m.jnt_solimp,
      m.jnt_pos,
      m.jnt_axis,
      m.jnt_stiffness,
      m.jnt_range,
      m.jnt_actfrcrange,
      m.jnt_margin,
      m._impl.jnt_limited_slide_hinge_adr,
      m._impl.jnt_limited_ball_adr,
      m.jnt_actgravcomp,
      m.dof_bodyid,
      m.dof_jntid,
      m.dof_parentid,
      m.dof_Madr,
      m.dof_armature,
      m.dof_damping,
      m.dof_invweight0,
      m.dof_frictionloss,
      m.dof_solimp,
      m.dof_solref,
      m._impl.dof_tri_row,
      m._impl.dof_tri_col,
      m.geom_type,
      m.geom_condim,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_rbound,
      m.geom_pos,
      m.geom_quat,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.site_type,
      m.site_bodyid,
      m.site_size,
      m.site_pos,
      m.site_quat,
      m.cam_mode,
      m.cam_bodyid,
      m.cam_targetbodyid,
      m.cam_pos,
      m.cam_quat,
      m.cam_poscom0,
      m.cam_pos0,
      m.cam_mat0,
      m.cam_fovy,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.light_mode,
      m._impl.light_bodyid,
      m._impl.light_targetbodyid,
      m.light_pos,
      m.light_dir,
      m.light_poscom0,
      m.light_pos0,
      m.light_dir0,
      m._impl.flex_dim,
      m._impl.flex_vertadr,
      m._impl.flex_edgeadr,
      m._impl.flex_elemedgeadr,
      m._impl.flex_vertbodyid,
      m._impl.flex_elem,
      m._impl.flex_elemedge,
      m._impl.flexedge_length0,
      m._impl.flex_stiffness,
      m._impl.flex_damping,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_vert,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.eq_solref,
      m.eq_solimp,
      m.eq_data,
      m._impl.eq_connect_adr,
      m._impl.eq_wld_adr,
      m._impl.eq_jnt_adr,
      m._impl.eq_ten_adr,
      m._impl.actuator_moment_tiles_nv,
      m._impl.actuator_moment_tiles_nu,
      m.actuator_trntype,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_trnid,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_ctrllimited,
      m.actuator_forcelimited,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      m.actuator_gear,
      m._impl.nxn_geom_pair,
      m._impl.nxn_pairid,
      m.pair_dim,
      m.pair_solref,
      m.pair_solreffriction,
      m.pair_solimp,
      m.pair_margin,
      m.pair_gap,
      m.pair_friction,
      m._impl.condim_max,
      m.tendon_adr,
      m.tendon_num,
      m._impl.tendon_limited_adr,
      m.tendon_solref_lim,
      m.tendon_solimp_lim,
      m.tendon_solref_fri,
      m.tendon_solimp_fri,
      m.tendon_range,
      m.tendon_margin,
      m.tendon_stiffness,
      m.tendon_damping,
      m.tendon_frictionloss,
      m.tendon_lengthspring,
      m.tendon_length0,
      m.tendon_invweight0,
      m.wrap_objid,
      m.wrap_prm,
      m.wrap_type,
      m._impl.tendon_jnt_adr,
      m._impl.tendon_site_pair_adr,
      m._impl.ten_wrapadr_site,
      m._impl.ten_wrapnum_site,
      m._impl.wrap_jnt_adr,
      m._impl.wrap_site_adr,
      m._impl.wrap_site_pair_adr,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m._impl.sensor_pos_adr,
      m._impl.sensor_vel_adr,
      m._impl.sensor_acc_adr,
      m._impl.sensor_touch_adr,
      m._impl.sensor_subtree_vel,
      m._impl.sensor_rne_postconstraint,
      m._impl.mocap_bodyid,
      d._impl.nconmax,
      d._impl.njmax,
      d.time,
      d.qpos,
      d.qvel,
      d.act,
      d.qacc_warmstart,
      d.ctrl,
      d.qfrc_applied,
      d.xfrc_applied,
      d.eq_active,
      d.mocap_pos,
      d.mocap_quat,
      d._impl.flexvert_xpos,
      d._impl.flexedge_length,
      d._impl.flexedge_velocity,
      d._impl.qLD,
      d._impl.qLDiagInv,
      d.qfrc_fluid,
      d.qacc_smooth,
      d._impl.efc__mv,
      d._impl.ncon,
      d._impl.ne,
      d._impl.ne_connect,
      d._impl.ne_weld,
      d._impl.ne_jnt,
      d._impl.ne_ten,
      d._impl.nf,
      d._impl.nl,
      d._impl.nefc,
      d._impl.fluid_applied,
      d.qacc,
      d.act_dot,
      d.xpos,
      d.xquat,
      d.xmat,
      d.xipos,
      d.ximat,
      d.xanchor,
      d.xaxis,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d._impl.light_xpos,
      d._impl.light_xdir,
      d.subtree_com,
      d._impl.cdof,
      d._impl.cinert,
      d._impl.actuator_length,
      d._impl.actuator_moment,
      d._impl.crb,
      d._impl.qM,
      d._impl.ten_velocity,
      d._impl.actuator_velocity,
      d.cvel,
      d._impl.cdof_dot,
      d.qfrc_bias,
      d._impl.qfrc_spring,
      d._impl.qfrc_damper,
      d.qfrc_gravcomp,
      d.qfrc_passive,
      d._impl.subtree_linvel,
      d._impl.subtree_angmom,
      d._impl.subtree_bodyvel,
      d.actuator_force,
      d.qfrc_actuator,
      d.qfrc_smooth,
      d.qfrc_constraint,
      d._impl.contact__dist,
      d._impl.contact__pos,
      d._impl.contact__frame,
      d._impl.contact__includemargin,
      d._impl.contact__friction,
      d._impl.contact__solref,
      d._impl.contact__solreffriction,
      d._impl.contact__solimp,
      d._impl.contact__dim,
      d._impl.contact__geom,
      d._impl.contact__efc_address,
      d._impl.contact__worldid,
      d._impl.efc__worldid,
      d._impl.efc__id,
      d._impl.efc__J,
      d._impl.efc__pos,
      d._impl.efc__margin,
      d._impl.efc__D,
      d._impl.efc__aref,
      d._impl.efc__frictionloss,
      d._impl.efc__force,
      d._impl.efc__Jaref,
      d._impl.efc__Ma,
      d._impl.efc__grad,
      d._impl.efc__grad_dot,
      d._impl.efc__Mgrad,
      d._impl.efc__search,
      d._impl.efc__search_dot,
      d._impl.efc__gauss,
      d._impl.efc__cost,
      d._impl.efc__prev_cost,
      d._impl.efc__solver_niter,
      d._impl.efc__active,
      d._impl.efc__gtol,
      d._impl.efc__jv,
      d._impl.efc__quad,
      d._impl.efc__quad_gauss,
      d._impl.efc__h,
      d._impl.efc__alpha,
      d._impl.efc__prev_grad,
      d._impl.efc__prev_Mgrad,
      d._impl.efc__beta,
      d._impl.efc__beta_num,
      d._impl.efc__beta_den,
      d._impl.efc__done,
      d._impl.efc__ls_done,
      d._impl.efc__p0,
      d._impl.efc__lo,
      d._impl.efc__lo_alpha,
      d._impl.efc__hi,
      d._impl.efc__hi_alpha,
      d._impl.efc__lo_next,
      d._impl.efc__lo_next_alpha,
      d._impl.efc__hi_next,
      d._impl.efc__hi_next_alpha,
      d._impl.efc__mid,
      d._impl.efc__mid_alpha,
      d._impl.efc__cost_candidate,
      d._impl.efc__quad_total_candidate,
      d._impl.efc__u,
      d._impl.efc__uu,
      d._impl.efc__uv,
      d._impl.efc__vv,
      d._impl.efc__condim,
      d._impl.collision_pair,
      d._impl.collision_pairid,
      d._impl.collision_worldid,
      d._impl.ncollision,
      d._impl.cacc,
      d._impl.cfrc_int,
      d._impl.cfrc_ext,
      d._impl.ten_length,
      d._impl.ten_J,
      d._impl.ten_wrapadr,
      d._impl.ten_wrapnum,
      d._impl.wrap_obj,
      d._impl.wrap_xpos,
      d.sensordata,
  )
  d = d.tree_replace({
      'qacc': out[0],
      'act_dot': out[1],
      'xpos': out[2],
      'xquat': out[3],
      'xmat': out[4],
      'xipos': out[5],
      'ximat': out[6],
      'xanchor': out[7],
      'xaxis': out[8],
      'geom_xpos': out[9],
      'geom_xmat': out[10],
      'site_xpos': out[11],
      'site_xmat': out[12],
      'cam_xpos': out[13],
      'cam_xmat': out[14],
      'subtree_com': out[15],
      'cvel': out[16],
      'qfrc_bias': out[17],
      'qfrc_gravcomp': out[18],
      'qfrc_passive': out[19],
      'actuator_force': out[20],
      'qfrc_actuator': out[21],
      'qfrc_smooth': out[22],
      'qfrc_constraint': out[23],
      'sensordata': out[24],
  })
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def forward(m: types.Model, d: types.Data):
  return _forward_jax_impl(m, d)


@forward.def_vmap
@ffi.marshal_custom_vmap
def forward_vmap(unused_axis_size, is_batched, m, d):
  d = forward(m, d)
  return d, is_batched[1]


@ffi.format_args_for_warp
def _step_shim(
    # Model
    nworld: int,
    nv: int,
    nu: int,
    na: int,
    nbody: int,
    njnt: int,
    ngeom: int,
    nsite: int,
    ncam: int,
    nlight: int,
    nflexelem: int,
    neq: int,
    nmocap: int,
    ngravcomp: int,
    nM: int,
    ntendon: int,
    nlsp: int,
    opt__timestep: float,
    opt__impratio: float,
    opt__tolerance: float,
    opt__ls_tolerance: float,
    opt__gravity: wp.array(dtype=wp.vec3),
    opt__integrator: int,
    opt__cone: int,
    opt__solver: int,
    opt__iterations: int,
    opt__ls_iterations: int,
    opt__disableflags: int,
    opt__is_sparse: bool,
    opt__ls_parallel: bool,
    opt__wind: wp.array(dtype=wp.vec3),
    opt__has_wind: bool,
    opt__density: float,
    opt__viscosity: float,
    stat__meaninertia: float,
    qpos0: wp.array2d(dtype=float),
    qpos_spring: wp.array2d(dtype=float),
    qM_fullm_i: wp.array(dtype=int),
    qM_fullm_j: wp.array(dtype=int),
    qM_mulm_i: wp.array(dtype=int),
    qM_mulm_j: wp.array(dtype=int),
    qM_madr_ij: wp.array(dtype=int),
    qLD_updates: tuple[wp.array(dtype=wp.vec3i), ...],
    M_rownnz: wp.array(dtype=int),
    M_rowadr: wp.array(dtype=int),
    mapM2M: wp.array(dtype=int),
    qM_tiles: tuple[mjwarp_types.TileSet, ...],
    body_tree: tuple[wp.array(dtype=int), ...],
    body_parentid: wp.array(dtype=int),
    body_rootid: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_dofnum: wp.array(dtype=int),
    body_dofadr: wp.array(dtype=int),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    body_mass: wp.array2d(dtype=float),
    body_subtreemass: wp.array2d(dtype=float),
    subtree_mass: wp.array2d(dtype=float),
    body_inertia: wp.array2d(dtype=wp.vec3),
    body_invweight0: wp.array3d(dtype=float),
    body_gravcomp: wp.array2d(dtype=float),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_dofadr: wp.array(dtype=int),
    jnt_bodyid: wp.array(dtype=int),
    jnt_actfrclimited: wp.array(dtype=bool),
    jnt_solref: wp.array2d(dtype=wp.vec2),
    jnt_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    jnt_stiffness: wp.array2d(dtype=float),
    jnt_range: wp.array3d(dtype=float),
    jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
    jnt_margin: wp.array2d(dtype=float),
    jnt_limited_slide_hinge_adr: wp.array(dtype=int),
    jnt_limited_ball_adr: wp.array(dtype=int),
    jnt_actgravcomp: wp.array(dtype=int),
    dof_bodyid: wp.array(dtype=int),
    dof_jntid: wp.array(dtype=int),
    dof_parentid: wp.array(dtype=int),
    dof_Madr: wp.array(dtype=int),
    dof_armature: wp.array2d(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    dof_invweight0: wp.array2d(dtype=float),
    dof_frictionloss: wp.array2d(dtype=float),
    dof_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    dof_solref: wp.array2d(dtype=wp.vec2),
    dof_tri_row: wp.array(dtype=int),
    dof_tri_col: wp.array(dtype=int),
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_bodyid: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    site_type: wp.array(dtype=int),
    site_bodyid: wp.array(dtype=int),
    site_size: wp.array(dtype=wp.vec3),
    site_pos: wp.array2d(dtype=wp.vec3),
    site_quat: wp.array2d(dtype=wp.quat),
    cam_mode: wp.array(dtype=int),
    cam_bodyid: wp.array(dtype=int),
    cam_targetbodyid: wp.array(dtype=int),
    cam_pos: wp.array2d(dtype=wp.vec3),
    cam_quat: wp.array2d(dtype=wp.quat),
    cam_poscom0: wp.array2d(dtype=wp.vec3),
    cam_pos0: wp.array2d(dtype=wp.vec3),
    cam_mat0: wp.array2d(dtype=wp.mat33),
    cam_fovy: wp.array(dtype=float),
    cam_resolution: wp.array(dtype=wp.vec2i),
    cam_sensorsize: wp.array(dtype=wp.vec2),
    cam_intrinsic: wp.array(dtype=wp.vec4),
    light_mode: wp.array(dtype=int),
    light_bodyid: wp.array(dtype=int),
    light_targetbodyid: wp.array(dtype=int),
    light_pos: wp.array2d(dtype=wp.vec3),
    light_dir: wp.array2d(dtype=wp.vec3),
    light_poscom0: wp.array2d(dtype=wp.vec3),
    light_pos0: wp.array2d(dtype=wp.vec3),
    light_dir0: wp.array2d(dtype=wp.vec3),
    flex_dim: wp.array(dtype=int),
    flex_vertadr: wp.array(dtype=int),
    flex_edgeadr: wp.array(dtype=int),
    flex_elemedgeadr: wp.array(dtype=int),
    flex_vertbodyid: wp.array(dtype=int),
    flex_elem: wp.array(dtype=int),
    flex_elemedge: wp.array(dtype=int),
    flexedge_length0: wp.array(dtype=float),
    flex_stiffness: wp.array(dtype=float),
    flex_damping: wp.array(dtype=float),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    eq_obj1id: wp.array(dtype=int),
    eq_obj2id: wp.array(dtype=int),
    eq_objtype: wp.array(dtype=int),
    eq_solref: wp.array2d(dtype=wp.vec2),
    eq_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    eq_data: wp.array2d(dtype=mjwarp_types.vec11),
    eq_connect_adr: wp.array(dtype=int),
    eq_wld_adr: wp.array(dtype=int),
    eq_jnt_adr: wp.array(dtype=int),
    eq_ten_adr: wp.array(dtype=int),
    actuator_moment_tiles_nv: tuple[mjwarp_types.TileSet, ...],
    actuator_moment_tiles_nu: tuple[mjwarp_types.TileSet, ...],
    actuator_affine_bias_gain: bool,
    actuator_trntype: wp.array(dtype=int),
    actuator_dyntype: wp.array(dtype=int),
    actuator_gaintype: wp.array(dtype=int),
    actuator_biastype: wp.array(dtype=int),
    actuator_trnid: wp.array(dtype=wp.vec2i),
    actuator_actadr: wp.array(dtype=int),
    actuator_actnum: wp.array(dtype=int),
    actuator_ctrllimited: wp.array(dtype=bool),
    actuator_forcelimited: wp.array(dtype=bool),
    actuator_actlimited: wp.array(dtype=bool),
    actuator_dynprm: wp.array2d(dtype=mjwarp_types.vec10f),
    actuator_gainprm: wp.array2d(dtype=mjwarp_types.vec10f),
    actuator_biasprm: wp.array2d(dtype=mjwarp_types.vec10f),
    actuator_ctrlrange: wp.array2d(dtype=wp.vec2),
    actuator_forcerange: wp.array2d(dtype=wp.vec2),
    actuator_actrange: wp.array2d(dtype=wp.vec2),
    actuator_gear: wp.array2d(dtype=wp.spatial_vector),
    nxn_geom_pair: wp.array(dtype=wp.vec2i),
    nxn_pairid: wp.array(dtype=int),
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=mjwarp_types.vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=mjwarp_types.vec5),
    condim_max: int,
    tendon_adr: wp.array(dtype=int),
    tendon_num: wp.array(dtype=int),
    tendon_limited_adr: wp.array(dtype=int),
    tendon_solref_lim: wp.array2d(dtype=wp.vec2),
    tendon_solimp_lim: wp.array2d(dtype=mjwarp_types.vec5),
    tendon_solref_fri: wp.array2d(dtype=wp.vec2),
    tendon_solimp_fri: wp.array2d(dtype=mjwarp_types.vec5),
    tendon_range: wp.array2d(dtype=wp.vec2),
    tendon_margin: wp.array2d(dtype=float),
    tendon_stiffness: wp.array2d(dtype=float),
    tendon_damping: wp.array2d(dtype=float),
    tendon_frictionloss: wp.array2d(dtype=float),
    tendon_lengthspring: wp.array2d(dtype=wp.vec2),
    tendon_length0: wp.array2d(dtype=float),
    tendon_invweight0: wp.array2d(dtype=float),
    wrap_objid: wp.array(dtype=int),
    wrap_prm: wp.array(dtype=float),
    wrap_type: wp.array(dtype=int),
    tendon_jnt_adr: wp.array(dtype=int),
    tendon_site_pair_adr: wp.array(dtype=int),
    ten_wrapadr_site: wp.array(dtype=int),
    ten_wrapnum_site: wp.array(dtype=int),
    wrap_jnt_adr: wp.array(dtype=int),
    wrap_site_adr: wp.array(dtype=int),
    wrap_site_pair_adr: wp.array(dtype=int),
    sensor_type: wp.array(dtype=int),
    sensor_datatype: wp.array(dtype=int),
    sensor_objtype: wp.array(dtype=int),
    sensor_objid: wp.array(dtype=int),
    sensor_reftype: wp.array(dtype=int),
    sensor_refid: wp.array(dtype=int),
    sensor_adr: wp.array(dtype=int),
    sensor_cutoff: wp.array(dtype=float),
    sensor_pos_adr: wp.array(dtype=int),
    sensor_vel_adr: wp.array(dtype=int),
    sensor_acc_adr: wp.array(dtype=int),
    sensor_touch_adr: wp.array(dtype=int),
    sensor_subtree_vel: bool,
    sensor_rne_postconstraint: bool,
    mocap_bodyid: wp.array(dtype=int),
    # Data
    nconmax: int,
    njmax: int,
    qacc_warmstart: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    qfrc_applied: wp.array2d(dtype=float),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
    eq_active: wp.array2d(dtype=bool),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
    flexvert_xpos: wp.array2d(dtype=wp.vec3),
    flexedge_length: wp.array2d(dtype=float),
    flexedge_velocity: wp.array2d(dtype=float),
    qLD: wp.array3d(dtype=float),
    qLDiagInv: wp.array2d(dtype=float),
    qfrc_fluid: wp.array2d(dtype=float),
    qacc_smooth: wp.array2d(dtype=float),
    efc__mv: wp.array2d(dtype=float),
    qpos_t0: wp.array2d(dtype=float),
    qvel_t0: wp.array2d(dtype=float),
    act_t0: wp.array2d(dtype=float),
    qLD_integration: wp.array3d(dtype=float),
    qLDiagInv_integration: wp.array2d(dtype=float),
    ncon: wp.array(dtype=int),
    ne: wp.array(dtype=int),
    ne_connect: wp.array(dtype=int),
    ne_weld: wp.array(dtype=int),
    ne_jnt: wp.array(dtype=int),
    ne_ten: wp.array(dtype=int),
    nf: wp.array(dtype=int),
    nl: wp.array(dtype=int),
    nefc: wp.array(dtype=int),
    time: wp.array(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    fluid_applied: wp.array2d(dtype=wp.spatial_vector),
    qacc: wp.array2d(dtype=float),
    act_dot: wp.array2d(dtype=float),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    xmat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3),
    ximat: wp.array2d(dtype=wp.mat33),
    xanchor: wp.array2d(dtype=wp.vec3),
    xaxis: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    cinert: wp.array2d(dtype=mjwarp_types.vec10),
    actuator_length: wp.array2d(dtype=float),
    actuator_moment: wp.array3d(dtype=float),
    crb: wp.array2d(dtype=mjwarp_types.vec10),
    qM: wp.array3d(dtype=float),
    ten_velocity: wp.array2d(dtype=float),
    actuator_velocity: wp.array2d(dtype=float),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    cdof_dot: wp.array2d(dtype=wp.spatial_vector),
    qfrc_bias: wp.array2d(dtype=float),
    qfrc_spring: wp.array2d(dtype=float),
    qfrc_damper: wp.array2d(dtype=float),
    qfrc_gravcomp: wp.array2d(dtype=float),
    qfrc_passive: wp.array2d(dtype=float),
    subtree_linvel: wp.array2d(dtype=wp.vec3),
    subtree_angmom: wp.array2d(dtype=wp.vec3),
    subtree_bodyvel: wp.array2d(dtype=wp.spatial_vector),
    actuator_force: wp.array2d(dtype=float),
    qfrc_actuator: wp.array2d(dtype=float),
    qfrc_smooth: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
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
    contact__efc_address: wp.array2d(dtype=int),
    contact__worldid: wp.array(dtype=int),
    efc__worldid: wp.array(dtype=int),
    efc__id: wp.array(dtype=int),
    efc__J: wp.array2d(dtype=float),
    efc__pos: wp.array(dtype=float),
    efc__margin: wp.array(dtype=float),
    efc__D: wp.array(dtype=float),
    efc__aref: wp.array(dtype=float),
    efc__frictionloss: wp.array(dtype=float),
    efc__force: wp.array(dtype=float),
    efc__Jaref: wp.array(dtype=float),
    efc__Ma: wp.array2d(dtype=float),
    efc__grad: wp.array2d(dtype=float),
    efc__grad_dot: wp.array(dtype=float),
    efc__Mgrad: wp.array2d(dtype=float),
    efc__search: wp.array2d(dtype=float),
    efc__search_dot: wp.array(dtype=float),
    efc__gauss: wp.array(dtype=float),
    efc__cost: wp.array(dtype=float),
    efc__prev_cost: wp.array(dtype=float),
    efc__solver_niter: wp.array(dtype=int),
    efc__active: wp.array(dtype=bool),
    efc__gtol: wp.array(dtype=float),
    efc__jv: wp.array(dtype=float),
    efc__quad: wp.array(dtype=wp.vec3),
    efc__quad_gauss: wp.array(dtype=wp.vec3),
    efc__h: wp.array3d(dtype=float),
    efc__alpha: wp.array(dtype=float),
    efc__prev_grad: wp.array2d(dtype=float),
    efc__prev_Mgrad: wp.array2d(dtype=float),
    efc__beta: wp.array(dtype=float),
    efc__beta_num: wp.array(dtype=float),
    efc__beta_den: wp.array(dtype=float),
    efc__done: wp.array(dtype=bool),
    efc__ls_done: wp.array(dtype=bool),
    efc__p0: wp.array(dtype=wp.vec3),
    efc__lo: wp.array(dtype=wp.vec3),
    efc__lo_alpha: wp.array(dtype=float),
    efc__hi: wp.array(dtype=wp.vec3),
    efc__hi_alpha: wp.array(dtype=float),
    efc__lo_next: wp.array(dtype=wp.vec3),
    efc__lo_next_alpha: wp.array(dtype=float),
    efc__hi_next: wp.array(dtype=wp.vec3),
    efc__hi_next_alpha: wp.array(dtype=float),
    efc__mid: wp.array(dtype=wp.vec3),
    efc__mid_alpha: wp.array(dtype=float),
    efc__cost_candidate: wp.array2d(dtype=float),
    efc__quad_total_candidate: wp.array2d(dtype=wp.vec3),
    efc__u: wp.array(dtype=mjwarp_types.vec6),
    efc__uu: wp.array(dtype=float),
    efc__uv: wp.array(dtype=float),
    efc__vv: wp.array(dtype=float),
    efc__condim: wp.array(dtype=int),
    qvel_rk: wp.array2d(dtype=float),
    qacc_rk: wp.array2d(dtype=float),
    act_dot_rk: wp.array2d(dtype=float),
    qfrc_integration: wp.array2d(dtype=float),
    qacc_integration: wp.array2d(dtype=float),
    act_vel_integration: wp.array2d(dtype=float),
    qM_integration: wp.array3d(dtype=float),
    collision_pair: wp.array(dtype=wp.vec2i),
    collision_pairid: wp.array(dtype=int),
    collision_worldid: wp.array(dtype=int),
    ncollision: wp.array(dtype=int),
    cacc: wp.array2d(dtype=wp.spatial_vector),
    cfrc_int: wp.array2d(dtype=wp.spatial_vector),
    cfrc_ext: wp.array2d(dtype=wp.spatial_vector),
    ten_length: wp.array2d(dtype=float),
    ten_J: wp.array3d(dtype=float),
    ten_wrapadr: wp.array2d(dtype=int),
    ten_wrapnum: wp.array2d(dtype=int),
    wrap_obj: wp.array2d(dtype=wp.vec2i),
    wrap_xpos: wp.array2d(dtype=wp.spatial_vector),
    sensordata: wp.array2d(dtype=float),
):
  _m.stat = _s
  _m.opt = _o
  _d.efc = _e
  _d.contact = _c
  _m.M_rowadr = M_rowadr
  _m.M_rownnz = M_rownnz
  _m.actuator_actadr = actuator_actadr
  _m.actuator_actlimited = actuator_actlimited
  _m.actuator_actnum = actuator_actnum
  _m.actuator_actrange = actuator_actrange
  _m.actuator_affine_bias_gain = actuator_affine_bias_gain
  _m.actuator_biasprm = actuator_biasprm
  _m.actuator_biastype = actuator_biastype
  _m.actuator_ctrllimited = actuator_ctrllimited
  _m.actuator_ctrlrange = actuator_ctrlrange
  _m.actuator_dynprm = actuator_dynprm
  _m.actuator_dyntype = actuator_dyntype
  _m.actuator_forcelimited = actuator_forcelimited
  _m.actuator_forcerange = actuator_forcerange
  _m.actuator_gainprm = actuator_gainprm
  _m.actuator_gaintype = actuator_gaintype
  _m.actuator_gear = actuator_gear
  _m.actuator_moment_tiles_nu = actuator_moment_tiles_nu
  _m.actuator_moment_tiles_nv = actuator_moment_tiles_nv
  _m.actuator_trnid = actuator_trnid
  _m.actuator_trntype = actuator_trntype
  _m.body_dofadr = body_dofadr
  _m.body_dofnum = body_dofnum
  _m.body_gravcomp = body_gravcomp
  _m.body_inertia = body_inertia
  _m.body_invweight0 = body_invweight0
  _m.body_ipos = body_ipos
  _m.body_iquat = body_iquat
  _m.body_jntadr = body_jntadr
  _m.body_jntnum = body_jntnum
  _m.body_mass = body_mass
  _m.body_parentid = body_parentid
  _m.body_pos = body_pos
  _m.body_quat = body_quat
  _m.body_rootid = body_rootid
  _m.body_subtreemass = body_subtreemass
  _m.body_tree = body_tree
  _m.cam_bodyid = cam_bodyid
  _m.cam_fovy = cam_fovy
  _m.cam_intrinsic = cam_intrinsic
  _m.cam_mat0 = cam_mat0
  _m.cam_mode = cam_mode
  _m.cam_pos = cam_pos
  _m.cam_pos0 = cam_pos0
  _m.cam_poscom0 = cam_poscom0
  _m.cam_quat = cam_quat
  _m.cam_resolution = cam_resolution
  _m.cam_sensorsize = cam_sensorsize
  _m.cam_targetbodyid = cam_targetbodyid
  _m.condim_max = condim_max
  _m.dof_Madr = dof_Madr
  _m.dof_armature = dof_armature
  _m.dof_bodyid = dof_bodyid
  _m.dof_damping = dof_damping
  _m.dof_frictionloss = dof_frictionloss
  _m.dof_invweight0 = dof_invweight0
  _m.dof_jntid = dof_jntid
  _m.dof_parentid = dof_parentid
  _m.dof_solimp = dof_solimp
  _m.dof_solref = dof_solref
  _m.dof_tri_col = dof_tri_col
  _m.dof_tri_row = dof_tri_row
  _m.eq_connect_adr = eq_connect_adr
  _m.eq_data = eq_data
  _m.eq_jnt_adr = eq_jnt_adr
  _m.eq_obj1id = eq_obj1id
  _m.eq_obj2id = eq_obj2id
  _m.eq_objtype = eq_objtype
  _m.eq_solimp = eq_solimp
  _m.eq_solref = eq_solref
  _m.eq_ten_adr = eq_ten_adr
  _m.eq_wld_adr = eq_wld_adr
  _m.flex_damping = flex_damping
  _m.flex_dim = flex_dim
  _m.flex_edgeadr = flex_edgeadr
  _m.flex_elem = flex_elem
  _m.flex_elemedge = flex_elemedge
  _m.flex_elemedgeadr = flex_elemedgeadr
  _m.flex_stiffness = flex_stiffness
  _m.flex_vertadr = flex_vertadr
  _m.flex_vertbodyid = flex_vertbodyid
  _m.flexedge_length0 = flexedge_length0
  _m.geom_bodyid = geom_bodyid
  _m.geom_condim = geom_condim
  _m.geom_dataid = geom_dataid
  _m.geom_friction = geom_friction
  _m.geom_gap = geom_gap
  _m.geom_margin = geom_margin
  _m.geom_pos = geom_pos
  _m.geom_priority = geom_priority
  _m.geom_quat = geom_quat
  _m.geom_rbound = geom_rbound
  _m.geom_size = geom_size
  _m.geom_solimp = geom_solimp
  _m.geom_solmix = geom_solmix
  _m.geom_solref = geom_solref
  _m.geom_type = geom_type
  _m.jnt_actfrclimited = jnt_actfrclimited
  _m.jnt_actfrcrange = jnt_actfrcrange
  _m.jnt_actgravcomp = jnt_actgravcomp
  _m.jnt_axis = jnt_axis
  _m.jnt_bodyid = jnt_bodyid
  _m.jnt_dofadr = jnt_dofadr
  _m.jnt_limited_ball_adr = jnt_limited_ball_adr
  _m.jnt_limited_slide_hinge_adr = jnt_limited_slide_hinge_adr
  _m.jnt_margin = jnt_margin
  _m.jnt_pos = jnt_pos
  _m.jnt_qposadr = jnt_qposadr
  _m.jnt_range = jnt_range
  _m.jnt_solimp = jnt_solimp
  _m.jnt_solref = jnt_solref
  _m.jnt_stiffness = jnt_stiffness
  _m.jnt_type = jnt_type
  _m.light_bodyid = light_bodyid
  _m.light_dir = light_dir
  _m.light_dir0 = light_dir0
  _m.light_mode = light_mode
  _m.light_pos = light_pos
  _m.light_pos0 = light_pos0
  _m.light_poscom0 = light_poscom0
  _m.light_targetbodyid = light_targetbodyid
  _m.mapM2M = mapM2M
  _m.mesh_vert = mesh_vert
  _m.mesh_vertadr = mesh_vertadr
  _m.mesh_vertnum = mesh_vertnum
  _m.mocap_bodyid = mocap_bodyid
  _m.nM = nM
  _m.na = na
  _m.nbody = nbody
  _m.ncam = ncam
  _m.neq = neq
  _m.nflexelem = nflexelem
  _m.ngeom = ngeom
  _m.ngravcomp = ngravcomp
  _m.njnt = njnt
  _m.nlight = nlight
  _m.nlsp = nlsp
  _m.nmocap = nmocap
  _m.nsite = nsite
  _m.ntendon = ntendon
  _m.nu = nu
  _m.nv = nv
  _m.nxn_geom_pair = nxn_geom_pair
  _m.nxn_pairid = nxn_pairid
  _m.opt.cone = opt__cone
  _m.opt.density = opt__density
  _m.opt.disableflags = opt__disableflags
  _m.opt.gravity = opt__gravity
  _m.opt.has_wind = opt__has_wind
  _m.opt.impratio = opt__impratio
  _m.opt.integrator = opt__integrator
  _m.opt.is_sparse = opt__is_sparse
  _m.opt.iterations = opt__iterations
  _m.opt.ls_iterations = opt__ls_iterations
  _m.opt.ls_parallel = opt__ls_parallel
  _m.opt.ls_tolerance = opt__ls_tolerance
  _m.opt.solver = opt__solver
  _m.opt.timestep = opt__timestep
  _m.opt.tolerance = opt__tolerance
  _m.opt.viscosity = opt__viscosity
  _m.opt.wind = opt__wind
  _m.pair_dim = pair_dim
  _m.pair_friction = pair_friction
  _m.pair_gap = pair_gap
  _m.pair_margin = pair_margin
  _m.pair_solimp = pair_solimp
  _m.pair_solref = pair_solref
  _m.pair_solreffriction = pair_solreffriction
  _m.qLD_updates = qLD_updates
  _m.qM_fullm_i = qM_fullm_i
  _m.qM_fullm_j = qM_fullm_j
  _m.qM_madr_ij = qM_madr_ij
  _m.qM_mulm_i = qM_mulm_i
  _m.qM_mulm_j = qM_mulm_j
  _m.qM_tiles = qM_tiles
  _m.qpos0 = qpos0
  _m.qpos_spring = qpos_spring
  _m.sensor_acc_adr = sensor_acc_adr
  _m.sensor_adr = sensor_adr
  _m.sensor_cutoff = sensor_cutoff
  _m.sensor_datatype = sensor_datatype
  _m.sensor_objid = sensor_objid
  _m.sensor_objtype = sensor_objtype
  _m.sensor_pos_adr = sensor_pos_adr
  _m.sensor_refid = sensor_refid
  _m.sensor_reftype = sensor_reftype
  _m.sensor_rne_postconstraint = sensor_rne_postconstraint
  _m.sensor_subtree_vel = sensor_subtree_vel
  _m.sensor_touch_adr = sensor_touch_adr
  _m.sensor_type = sensor_type
  _m.sensor_vel_adr = sensor_vel_adr
  _m.site_bodyid = site_bodyid
  _m.site_pos = site_pos
  _m.site_quat = site_quat
  _m.site_size = site_size
  _m.site_type = site_type
  _m.stat.meaninertia = stat__meaninertia
  _m.subtree_mass = subtree_mass
  _m.ten_wrapadr_site = ten_wrapadr_site
  _m.ten_wrapnum_site = ten_wrapnum_site
  _m.tendon_adr = tendon_adr
  _m.tendon_damping = tendon_damping
  _m.tendon_frictionloss = tendon_frictionloss
  _m.tendon_invweight0 = tendon_invweight0
  _m.tendon_jnt_adr = tendon_jnt_adr
  _m.tendon_length0 = tendon_length0
  _m.tendon_lengthspring = tendon_lengthspring
  _m.tendon_limited_adr = tendon_limited_adr
  _m.tendon_margin = tendon_margin
  _m.tendon_num = tendon_num
  _m.tendon_range = tendon_range
  _m.tendon_site_pair_adr = tendon_site_pair_adr
  _m.tendon_solimp_fri = tendon_solimp_fri
  _m.tendon_solimp_lim = tendon_solimp_lim
  _m.tendon_solref_fri = tendon_solref_fri
  _m.tendon_solref_lim = tendon_solref_lim
  _m.tendon_stiffness = tendon_stiffness
  _m.wrap_jnt_adr = wrap_jnt_adr
  _m.wrap_objid = wrap_objid
  _m.wrap_prm = wrap_prm
  _m.wrap_site_adr = wrap_site_adr
  _m.wrap_site_pair_adr = wrap_site_pair_adr
  _m.wrap_type = wrap_type
  _d.act = act
  _d.act_dot = act_dot
  _d.act_dot_rk = act_dot_rk
  _d.act_t0 = act_t0
  _d.act_vel_integration = act_vel_integration
  _d.actuator_force = actuator_force
  _d.actuator_length = actuator_length
  _d.actuator_moment = actuator_moment
  _d.actuator_velocity = actuator_velocity
  _d.cacc = cacc
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.cdof = cdof
  _d.cdof_dot = cdof_dot
  _d.cfrc_ext = cfrc_ext
  _d.cfrc_int = cfrc_int
  _d.cinert = cinert
  _d.collision_pair = collision_pair
  _d.collision_pairid = collision_pairid
  _d.collision_worldid = collision_worldid
  _d.contact.dim = contact__dim
  _d.contact.dist = contact__dist
  _d.contact.efc_address = contact__efc_address
  _d.contact.frame = contact__frame
  _d.contact.friction = contact__friction
  _d.contact.geom = contact__geom
  _d.contact.includemargin = contact__includemargin
  _d.contact.pos = contact__pos
  _d.contact.solimp = contact__solimp
  _d.contact.solref = contact__solref
  _d.contact.solreffriction = contact__solreffriction
  _d.contact.worldid = contact__worldid
  _d.crb = crb
  _d.ctrl = ctrl
  _d.cvel = cvel
  _d.efc.D = efc__D
  _d.efc.J = efc__J
  _d.efc.Jaref = efc__Jaref
  _d.efc.Ma = efc__Ma
  _d.efc.Mgrad = efc__Mgrad
  _d.efc.active = efc__active
  _d.efc.alpha = efc__alpha
  _d.efc.aref = efc__aref
  _d.efc.beta = efc__beta
  _d.efc.beta_den = efc__beta_den
  _d.efc.beta_num = efc__beta_num
  _d.efc.condim = efc__condim
  _d.efc.cost = efc__cost
  _d.efc.cost_candidate = efc__cost_candidate
  _d.efc.done = efc__done
  _d.efc.force = efc__force
  _d.efc.frictionloss = efc__frictionloss
  _d.efc.gauss = efc__gauss
  _d.efc.grad = efc__grad
  _d.efc.grad_dot = efc__grad_dot
  _d.efc.gtol = efc__gtol
  _d.efc.h = efc__h
  _d.efc.hi = efc__hi
  _d.efc.hi_alpha = efc__hi_alpha
  _d.efc.hi_next = efc__hi_next
  _d.efc.hi_next_alpha = efc__hi_next_alpha
  _d.efc.id = efc__id
  _d.efc.jv = efc__jv
  _d.efc.lo = efc__lo
  _d.efc.lo_alpha = efc__lo_alpha
  _d.efc.lo_next = efc__lo_next
  _d.efc.lo_next_alpha = efc__lo_next_alpha
  _d.efc.ls_done = efc__ls_done
  _d.efc.margin = efc__margin
  _d.efc.mid = efc__mid
  _d.efc.mid_alpha = efc__mid_alpha
  _d.efc.mv = efc__mv
  _d.efc.p0 = efc__p0
  _d.efc.pos = efc__pos
  _d.efc.prev_Mgrad = efc__prev_Mgrad
  _d.efc.prev_cost = efc__prev_cost
  _d.efc.prev_grad = efc__prev_grad
  _d.efc.quad = efc__quad
  _d.efc.quad_gauss = efc__quad_gauss
  _d.efc.quad_total_candidate = efc__quad_total_candidate
  _d.efc.search = efc__search
  _d.efc.search_dot = efc__search_dot
  _d.efc.solver_niter = efc__solver_niter
  _d.efc.u = efc__u
  _d.efc.uu = efc__uu
  _d.efc.uv = efc__uv
  _d.efc.vv = efc__vv
  _d.efc.worldid = efc__worldid
  _d.eq_active = eq_active
  _d.flexedge_length = flexedge_length
  _d.flexedge_velocity = flexedge_velocity
  _d.flexvert_xpos = flexvert_xpos
  _d.fluid_applied = fluid_applied
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  _d.mocap_pos = mocap_pos
  _d.mocap_quat = mocap_quat
  _d.ncollision = ncollision
  _d.ncon = ncon
  _d.nconmax = nconmax
  _d.ne = ne
  _d.ne_connect = ne_connect
  _d.ne_jnt = ne_jnt
  _d.ne_ten = ne_ten
  _d.ne_weld = ne_weld
  _d.nefc = nefc
  _d.nf = nf
  _d.njmax = njmax
  _d.nl = nl
  _d.qLD = qLD
  _d.qLD_integration = qLD_integration
  _d.qLDiagInv = qLDiagInv
  _d.qLDiagInv_integration = qLDiagInv_integration
  _d.qM = qM
  _d.qM_integration = qM_integration
  _d.qacc = qacc
  _d.qacc_integration = qacc_integration
  _d.qacc_rk = qacc_rk
  _d.qacc_smooth = qacc_smooth
  _d.qacc_warmstart = qacc_warmstart
  _d.qfrc_actuator = qfrc_actuator
  _d.qfrc_applied = qfrc_applied
  _d.qfrc_bias = qfrc_bias
  _d.qfrc_constraint = qfrc_constraint
  _d.qfrc_damper = qfrc_damper
  _d.qfrc_fluid = qfrc_fluid
  _d.qfrc_gravcomp = qfrc_gravcomp
  _d.qfrc_integration = qfrc_integration
  _d.qfrc_passive = qfrc_passive
  _d.qfrc_smooth = qfrc_smooth
  _d.qfrc_spring = qfrc_spring
  _d.qpos = qpos
  _d.qpos_t0 = qpos_t0
  _d.qvel = qvel
  _d.qvel_rk = qvel_rk
  _d.qvel_t0 = qvel_t0
  _d.sensordata = sensordata
  _d.site_xmat = site_xmat
  _d.site_xpos = site_xpos
  _d.subtree_angmom = subtree_angmom
  _d.subtree_bodyvel = subtree_bodyvel
  _d.subtree_com = subtree_com
  _d.subtree_linvel = subtree_linvel
  _d.ten_J = ten_J
  _d.ten_length = ten_length
  _d.ten_velocity = ten_velocity
  _d.ten_wrapadr = ten_wrapadr
  _d.ten_wrapnum = ten_wrapnum
  _d.time = time
  _d.wrap_obj = wrap_obj
  _d.wrap_xpos = wrap_xpos
  _d.xanchor = xanchor
  _d.xaxis = xaxis
  _d.xfrc_applied = xfrc_applied
  _d.ximat = ximat
  _d.xipos = xipos
  _d.xmat = xmat
  _d.xpos = xpos
  _d.xquat = xquat
  _d.nworld = nworld
  mjwarp.step(_m, _d)


def _step_jax_impl(m: types.Model, d: types.Data):
  output_dims = {
      'time': d.time.shape,
      'qpos': d.qpos.shape,
      'qvel': d.qvel.shape,
      'act': d.act.shape,
      'qacc': d.qacc.shape,
      'act_dot': d.act_dot.shape,
      'xpos': d.xpos.shape,
      'xquat': d.xquat.shape,
      'xmat': d.xmat.shape,
      'xipos': d.xipos.shape,
      'ximat': d.ximat.shape,
      'xanchor': d.xanchor.shape,
      'xaxis': d.xaxis.shape,
      'geom_xpos': d.geom_xpos.shape,
      'geom_xmat': d.geom_xmat.shape,
      'site_xpos': d.site_xpos.shape,
      'site_xmat': d.site_xmat.shape,
      'cam_xpos': d.cam_xpos.shape,
      'cam_xmat': d.cam_xmat.shape,
      'subtree_com': d.subtree_com.shape,
      'cvel': d.cvel.shape,
      'qfrc_bias': d.qfrc_bias.shape,
      'qfrc_gravcomp': d.qfrc_gravcomp.shape,
      'qfrc_passive': d.qfrc_passive.shape,
      'actuator_force': d.actuator_force.shape,
      'qfrc_actuator': d.qfrc_actuator.shape,
      'qfrc_smooth': d.qfrc_smooth.shape,
      'qfrc_constraint': d.qfrc_constraint.shape,
      'sensordata': d.sensordata.shape,
  }

  jf = ffi.jax_callable_variadic_tuple(
      _step_shim,
      num_outputs=29,
      output_dims=output_dims,
      vmap_method=None,
      graph_compatible=True,
      in_out_argnames={
          'time',
          'qpos',
          'qvel',
          'act',
          'qacc',
          'act_dot',
          'xpos',
          'xquat',
          'xmat',
          'xipos',
          'ximat',
          'xanchor',
          'xaxis',
          'geom_xpos',
          'geom_xmat',
          'site_xpos',
          'site_xmat',
          'cam_xpos',
          'cam_xmat',
          'subtree_com',
          'cvel',
          'qfrc_bias',
          'qfrc_gravcomp',
          'qfrc_passive',
          'actuator_force',
          'qfrc_actuator',
          'qfrc_smooth',
          'qfrc_constraint',
          'sensordata',
      },
  )
  out = jf(
      d.qpos.shape[0],
      m.nv,
      m.nu,
      m.na,
      m.nbody,
      m.njnt,
      m.ngeom,
      m.nsite,
      m.ncam,
      m.nlight,
      m._impl.nflexelem,
      m.neq,
      m.nmocap,
      m.ngravcomp,
      m.nM,
      m.ntendon,
      m._impl.nlsp,
      m.opt.timestep,
      m.opt.impratio,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.gravity,
      m.opt.integrator,
      m.opt.cone,
      m.opt.solver,
      m.opt.iterations,
      m.opt.ls_iterations,
      m.opt.disableflags,
      m.opt.is_sparse,
      m.opt.ls_parallel,
      m.opt.wind,
      m.opt.has_wind,
      m.opt.density,
      m.opt.viscosity,
      m.stat.meaninertia,
      m.qpos0,
      m.qpos_spring,
      m._impl.qM_fullm_i,
      m._impl.qM_fullm_j,
      m._impl.qM_mulm_i,
      m._impl.qM_mulm_j,
      m._impl.qM_madr_ij,
      m._impl.qLD_updates,
      m._impl.M_rownnz,
      m._impl.M_rowadr,
      m._impl.mapM2M,
      m._impl.qM_tiles,
      m._impl.body_tree,
      m.body_parentid,
      m.body_rootid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_dofnum,
      m.body_dofadr,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.body_mass,
      m.body_subtreemass,
      m._impl.subtree_mass,
      m.body_inertia,
      m.body_invweight0,
      m.body_gravcomp,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.jnt_actfrclimited,
      m.jnt_solref,
      m.jnt_solimp,
      m.jnt_pos,
      m.jnt_axis,
      m.jnt_stiffness,
      m.jnt_range,
      m.jnt_actfrcrange,
      m.jnt_margin,
      m._impl.jnt_limited_slide_hinge_adr,
      m._impl.jnt_limited_ball_adr,
      m.jnt_actgravcomp,
      m.dof_bodyid,
      m.dof_jntid,
      m.dof_parentid,
      m.dof_Madr,
      m.dof_armature,
      m.dof_damping,
      m.dof_invweight0,
      m.dof_frictionloss,
      m.dof_solimp,
      m.dof_solref,
      m._impl.dof_tri_row,
      m._impl.dof_tri_col,
      m.geom_type,
      m.geom_condim,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_rbound,
      m.geom_pos,
      m.geom_quat,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.site_type,
      m.site_bodyid,
      m.site_size,
      m.site_pos,
      m.site_quat,
      m.cam_mode,
      m.cam_bodyid,
      m.cam_targetbodyid,
      m.cam_pos,
      m.cam_quat,
      m.cam_poscom0,
      m.cam_pos0,
      m.cam_mat0,
      m.cam_fovy,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.light_mode,
      m._impl.light_bodyid,
      m._impl.light_targetbodyid,
      m.light_pos,
      m.light_dir,
      m.light_poscom0,
      m.light_pos0,
      m.light_dir0,
      m._impl.flex_dim,
      m._impl.flex_vertadr,
      m._impl.flex_edgeadr,
      m._impl.flex_elemedgeadr,
      m._impl.flex_vertbodyid,
      m._impl.flex_elem,
      m._impl.flex_elemedge,
      m._impl.flexedge_length0,
      m._impl.flex_stiffness,
      m._impl.flex_damping,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_vert,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.eq_solref,
      m.eq_solimp,
      m.eq_data,
      m._impl.eq_connect_adr,
      m._impl.eq_wld_adr,
      m._impl.eq_jnt_adr,
      m._impl.eq_ten_adr,
      m._impl.actuator_moment_tiles_nv,
      m._impl.actuator_moment_tiles_nu,
      m._impl.actuator_affine_bias_gain,
      m.actuator_trntype,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_trnid,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_ctrllimited,
      m.actuator_forcelimited,
      m.actuator_actlimited,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      m.actuator_actrange,
      m.actuator_gear,
      m._impl.nxn_geom_pair,
      m._impl.nxn_pairid,
      m.pair_dim,
      m.pair_solref,
      m.pair_solreffriction,
      m.pair_solimp,
      m.pair_margin,
      m.pair_gap,
      m.pair_friction,
      m._impl.condim_max,
      m.tendon_adr,
      m.tendon_num,
      m._impl.tendon_limited_adr,
      m.tendon_solref_lim,
      m.tendon_solimp_lim,
      m.tendon_solref_fri,
      m.tendon_solimp_fri,
      m.tendon_range,
      m.tendon_margin,
      m.tendon_stiffness,
      m.tendon_damping,
      m.tendon_frictionloss,
      m.tendon_lengthspring,
      m.tendon_length0,
      m.tendon_invweight0,
      m.wrap_objid,
      m.wrap_prm,
      m.wrap_type,
      m._impl.tendon_jnt_adr,
      m._impl.tendon_site_pair_adr,
      m._impl.ten_wrapadr_site,
      m._impl.ten_wrapnum_site,
      m._impl.wrap_jnt_adr,
      m._impl.wrap_site_adr,
      m._impl.wrap_site_pair_adr,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m._impl.sensor_pos_adr,
      m._impl.sensor_vel_adr,
      m._impl.sensor_acc_adr,
      m._impl.sensor_touch_adr,
      m._impl.sensor_subtree_vel,
      m._impl.sensor_rne_postconstraint,
      m._impl.mocap_bodyid,
      d._impl.nconmax,
      d._impl.njmax,
      d.qacc_warmstart,
      d.ctrl,
      d.qfrc_applied,
      d.xfrc_applied,
      d.eq_active,
      d.mocap_pos,
      d.mocap_quat,
      d._impl.flexvert_xpos,
      d._impl.flexedge_length,
      d._impl.flexedge_velocity,
      d._impl.qLD,
      d._impl.qLDiagInv,
      d.qfrc_fluid,
      d.qacc_smooth,
      d._impl.efc__mv,
      d._impl.qpos_t0,
      d._impl.qvel_t0,
      d._impl.act_t0,
      d._impl.qLD_integration,
      d._impl.qLDiagInv_integration,
      d._impl.ncon,
      d._impl.ne,
      d._impl.ne_connect,
      d._impl.ne_weld,
      d._impl.ne_jnt,
      d._impl.ne_ten,
      d._impl.nf,
      d._impl.nl,
      d._impl.nefc,
      d.time,
      d.qpos,
      d.qvel,
      d.act,
      d._impl.fluid_applied,
      d.qacc,
      d.act_dot,
      d.xpos,
      d.xquat,
      d.xmat,
      d.xipos,
      d.ximat,
      d.xanchor,
      d.xaxis,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d._impl.light_xpos,
      d._impl.light_xdir,
      d.subtree_com,
      d._impl.cdof,
      d._impl.cinert,
      d._impl.actuator_length,
      d._impl.actuator_moment,
      d._impl.crb,
      d._impl.qM,
      d._impl.ten_velocity,
      d._impl.actuator_velocity,
      d.cvel,
      d._impl.cdof_dot,
      d.qfrc_bias,
      d._impl.qfrc_spring,
      d._impl.qfrc_damper,
      d.qfrc_gravcomp,
      d.qfrc_passive,
      d._impl.subtree_linvel,
      d._impl.subtree_angmom,
      d._impl.subtree_bodyvel,
      d.actuator_force,
      d.qfrc_actuator,
      d.qfrc_smooth,
      d.qfrc_constraint,
      d._impl.contact__dist,
      d._impl.contact__pos,
      d._impl.contact__frame,
      d._impl.contact__includemargin,
      d._impl.contact__friction,
      d._impl.contact__solref,
      d._impl.contact__solreffriction,
      d._impl.contact__solimp,
      d._impl.contact__dim,
      d._impl.contact__geom,
      d._impl.contact__efc_address,
      d._impl.contact__worldid,
      d._impl.efc__worldid,
      d._impl.efc__id,
      d._impl.efc__J,
      d._impl.efc__pos,
      d._impl.efc__margin,
      d._impl.efc__D,
      d._impl.efc__aref,
      d._impl.efc__frictionloss,
      d._impl.efc__force,
      d._impl.efc__Jaref,
      d._impl.efc__Ma,
      d._impl.efc__grad,
      d._impl.efc__grad_dot,
      d._impl.efc__Mgrad,
      d._impl.efc__search,
      d._impl.efc__search_dot,
      d._impl.efc__gauss,
      d._impl.efc__cost,
      d._impl.efc__prev_cost,
      d._impl.efc__solver_niter,
      d._impl.efc__active,
      d._impl.efc__gtol,
      d._impl.efc__jv,
      d._impl.efc__quad,
      d._impl.efc__quad_gauss,
      d._impl.efc__h,
      d._impl.efc__alpha,
      d._impl.efc__prev_grad,
      d._impl.efc__prev_Mgrad,
      d._impl.efc__beta,
      d._impl.efc__beta_num,
      d._impl.efc__beta_den,
      d._impl.efc__done,
      d._impl.efc__ls_done,
      d._impl.efc__p0,
      d._impl.efc__lo,
      d._impl.efc__lo_alpha,
      d._impl.efc__hi,
      d._impl.efc__hi_alpha,
      d._impl.efc__lo_next,
      d._impl.efc__lo_next_alpha,
      d._impl.efc__hi_next,
      d._impl.efc__hi_next_alpha,
      d._impl.efc__mid,
      d._impl.efc__mid_alpha,
      d._impl.efc__cost_candidate,
      d._impl.efc__quad_total_candidate,
      d._impl.efc__u,
      d._impl.efc__uu,
      d._impl.efc__uv,
      d._impl.efc__vv,
      d._impl.efc__condim,
      d._impl.qvel_rk,
      d._impl.qacc_rk,
      d._impl.act_dot_rk,
      d._impl.qfrc_integration,
      d._impl.qacc_integration,
      d._impl.act_vel_integration,
      d._impl.qM_integration,
      d._impl.collision_pair,
      d._impl.collision_pairid,
      d._impl.collision_worldid,
      d._impl.ncollision,
      d._impl.cacc,
      d._impl.cfrc_int,
      d._impl.cfrc_ext,
      d._impl.ten_length,
      d._impl.ten_J,
      d._impl.ten_wrapadr,
      d._impl.ten_wrapnum,
      d._impl.wrap_obj,
      d._impl.wrap_xpos,
      d.sensordata,
  )
  d = d.tree_replace({
      'time': out[0],
      'qpos': out[1],
      'qvel': out[2],
      'act': out[3],
      'qacc': out[4],
      'act_dot': out[5],
      'xpos': out[6],
      'xquat': out[7],
      'xmat': out[8],
      'xipos': out[9],
      'ximat': out[10],
      'xanchor': out[11],
      'xaxis': out[12],
      'geom_xpos': out[13],
      'geom_xmat': out[14],
      'site_xpos': out[15],
      'site_xmat': out[16],
      'cam_xpos': out[17],
      'cam_xmat': out[18],
      'subtree_com': out[19],
      'cvel': out[20],
      'qfrc_bias': out[21],
      'qfrc_gravcomp': out[22],
      'qfrc_passive': out[23],
      'actuator_force': out[24],
      'qfrc_actuator': out[25],
      'qfrc_smooth': out[26],
      'qfrc_constraint': out[27],
      'sensordata': out[28],
  })
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def step(m: types.Model, d: types.Data):
  return _step_jax_impl(m, d)


@step.def_vmap
@ffi.marshal_custom_vmap
def step_vmap(unused_axis_size, is_batched, m, d):
  d = step(m, d)
  return d, is_batched[1]
