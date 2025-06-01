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
"""MJX Warp types.

DO NOT EDIT. This file is auto-generated.
"""
import dataclasses
from typing import Tuple
import jax
from jax import tree_util
from jax._src.interpreters import batching
from mujoco.mjx._src import dataclasses as mjx_dataclasses
import numpy as np

PyTreeNode = mjx_dataclasses.PyTreeNode


@dataclasses.dataclass(frozen=True)
@tree_util.register_pytree_node_class
class TileSet:
  """Tiling configuration for decomposable block diagonal matrix."""

  adr: np.ndarray
  size: int

  def tree_flatten(self):
    children = (self.adr, self.size)
    return (children, None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    adr_unflattened, size_unflattened = children
    return cls(adr=adr_unflattened, size=size_unflattened)


class StatisticWarp(PyTreeNode):
  """Derived fields from Statistic."""

  meaninertia: float


class OptionWarp(PyTreeNode):
  """Derived fields from Option."""

  cone: int
  density: float
  depth_extension: float
  disableflags: int
  epa_exact_neg_distance: bool
  epa_iterations: int
  gjk_iterations: int
  gravity: jax.Array
  impratio: float
  integrator: int
  is_sparse: bool
  iterations: int
  ls_iterations: int
  ls_parallel: bool
  ls_tolerance: float
  solver: int
  timestep: float
  tolerance: float
  viscosity: float
  wind: jax.Array


class ModelWarp(PyTreeNode):
  """Derived fields from Model."""

  M_colind: np.ndarray
  M_rowadr: np.ndarray
  M_rownnz: np.ndarray
  actuator_affine_bias_gain: bool
  actuator_moment_tiles_nu: Tuple[TileSet, ...]
  actuator_moment_tiles_nv: Tuple[TileSet, ...]
  body_tree: Tuple[np.ndarray, ...]
  condim_max: int
  dof_tri_col: np.ndarray
  dof_tri_row: np.ndarray
  eq_connect_adr: np.ndarray
  eq_jnt_adr: np.ndarray
  eq_ten_adr: np.ndarray
  eq_wld_adr: np.ndarray
  flex_damping: jax.Array
  flex_dim: jax.Array
  flex_edge: jax.Array
  flex_edgeadr: jax.Array
  flex_elem: jax.Array
  flex_elemedge: jax.Array
  flex_elemedgeadr: jax.Array
  flex_stiffness: jax.Array
  flex_vertadr: jax.Array
  flex_vertbodyid: jax.Array
  flex_vertnum: jax.Array
  flexedge_length0: jax.Array
  jnt_limited_ball_adr: np.ndarray
  jnt_limited_slide_hinge_adr: np.ndarray
  light_bodyid: jax.Array
  light_targetbodyid: jax.Array
  mapM2M: jax.Array
  mocap_bodyid: np.ndarray
  nflex: int
  nflexedge: int
  nflexelem: int
  nflexelemdata: int
  nflexvert: int
  nlsp: int
  nxn_geom_pair: np.ndarray
  nxn_pairid: np.ndarray
  qLD_updates: Tuple[np.ndarray, ...]
  qM_fullm_i: np.ndarray
  qM_fullm_j: np.ndarray
  qM_madr_ij: np.ndarray
  qM_mulm_i: np.ndarray
  qM_mulm_j: np.ndarray
  qM_tiles: Tuple[TileSet, ...]
  sensor_acc_adr: np.ndarray
  sensor_pos_adr: np.ndarray
  sensor_rne_postconstraint: bool
  sensor_subtree_vel: bool
  sensor_touch_adr: np.ndarray
  sensor_vel_adr: np.ndarray
  subtree_mass: jax.Array
  ten_wrapadr_site: np.ndarray
  ten_wrapnum_site: np.ndarray
  tendon_jnt_adr: np.ndarray
  tendon_limited_adr: np.ndarray
  tendon_site_adr: np.ndarray
  tendon_site_pair_adr: np.ndarray
  wrap_jnt_adr: np.ndarray
  wrap_site_adr: np.ndarray
  wrap_site_pair_adr: np.ndarray


class DataWarp(PyTreeNode):
  """Derived fields from Data."""

  act_dot_rk: jax.Array
  act_t0: jax.Array
  act_vel_integration: jax.Array
  actuator_length: jax.Array
  actuator_moment: jax.Array
  actuator_velocity: jax.Array
  cacc: jax.Array
  cdof: jax.Array
  cdof_dot: jax.Array
  cfrc_ext: jax.Array
  cfrc_int: jax.Array
  cinert: jax.Array
  collision_pair: jax.Array
  collision_pairid: jax.Array
  collision_worldid: jax.Array
  contact__dim: jax.Array
  contact__dist: jax.Array
  contact__efc_address: jax.Array
  contact__frame: jax.Array
  contact__friction: jax.Array
  contact__geom: jax.Array
  contact__includemargin: jax.Array
  contact__pos: jax.Array
  contact__solimp: jax.Array
  contact__solref: jax.Array
  contact__solreffriction: jax.Array
  contact__worldid: jax.Array
  crb: jax.Array
  efc__D: jax.Array
  efc__J: jax.Array
  efc__Jaref: jax.Array
  efc__Ma: jax.Array
  efc__Mgrad: jax.Array
  efc__active: jax.Array
  efc__alpha: jax.Array
  efc__aref: jax.Array
  efc__beta: jax.Array
  efc__beta_den: jax.Array
  efc__beta_num: jax.Array
  efc__condim: jax.Array
  efc__cost: jax.Array
  efc__cost_candidate: jax.Array
  efc__done: jax.Array
  efc__force: jax.Array
  efc__frictionloss: jax.Array
  efc__gauss: jax.Array
  efc__grad: jax.Array
  efc__grad_dot: jax.Array
  efc__gtol: jax.Array
  efc__h: jax.Array
  efc__hi: jax.Array
  efc__hi_alpha: jax.Array
  efc__hi_next: jax.Array
  efc__hi_next_alpha: jax.Array
  efc__id: jax.Array
  efc__jv: jax.Array
  efc__lo: jax.Array
  efc__lo_alpha: jax.Array
  efc__lo_next: jax.Array
  efc__lo_next_alpha: jax.Array
  efc__ls_done: jax.Array
  efc__margin: jax.Array
  efc__mid: jax.Array
  efc__mid_alpha: jax.Array
  efc__mv: jax.Array
  efc__p0: jax.Array
  efc__pos: jax.Array
  efc__prev_Mgrad: jax.Array
  efc__prev_cost: jax.Array
  efc__prev_grad: jax.Array
  efc__quad: jax.Array
  efc__quad_gauss: jax.Array
  efc__quad_total_candidate: jax.Array
  efc__search: jax.Array
  efc__search_dot: jax.Array
  efc__solver_niter: jax.Array
  efc__u: jax.Array
  efc__uu: jax.Array
  efc__uv: jax.Array
  efc__vv: jax.Array
  efc__worldid: jax.Array
  energy: jax.Array
  flexedge_length: jax.Array
  flexedge_velocity: jax.Array
  flexvert_xpos: jax.Array
  fluid_applied: jax.Array
  light_xdir: jax.Array
  light_xpos: jax.Array
  ncollision: jax.Array
  ncon: jax.Array
  nconmax: int
  ne: jax.Array
  ne_connect: jax.Array
  ne_jnt: jax.Array
  ne_ten: jax.Array
  ne_weld: jax.Array
  nefc: jax.Array
  nf: jax.Array
  njmax: int
  nl: jax.Array
  nworld: int
  qLD: jax.Array
  qLD_integration: jax.Array
  qLDiagInv: jax.Array
  qLDiagInv_integration: jax.Array
  qM: jax.Array
  qM_integration: jax.Array
  qacc_integration: jax.Array
  qacc_rk: jax.Array
  qfrc_damper: jax.Array
  qfrc_integration: jax.Array
  qfrc_spring: jax.Array
  qpos_t0: jax.Array
  qvel_rk: jax.Array
  qvel_t0: jax.Array
  sap_cumulative_sum: jax.Array
  sap_projection_lower: jax.Array
  sap_projection_upper: jax.Array
  sap_range: jax.Array
  sap_segment_index: jax.Array
  sap_sort_index: jax.Array
  subtree_angmom: jax.Array
  subtree_bodyvel: jax.Array
  subtree_linvel: jax.Array
  ten_J: jax.Array
  ten_length: jax.Array
  ten_velocity: jax.Array
  ten_wrapadr: jax.Array
  ten_wrapnum: jax.Array
  wrap_obj: jax.Array
  wrap_xpos: jax.Array
  shape = property(lambda self: self.cacc.shape)


_DATA_NON_VMAP = {
    'efc__J',
    'contact__friction',
    'ncon',
    'contact__solref',
    'contact__dist',
    'contact__efc_address',
    'contact__frame',
    'contact__worldid',
    'efc__Jaref',
    'efc__uu',
    'efc__force',
    'efc__jv',
    'efc__u',
    'efc__active',
    'collision_worldid',
    'contact__pos',
    'efc__id',
    'efc__quad',
    'contact__includemargin',
    'nf',
    'ne',
    'nl',
    'ne_jnt',
    'ne_weld',
    'efc__condim',
    'ne_connect',
    'contact__solreffriction',
    'nefc',
    'collision_pairid',
    'efc__frictionloss',
    'efc__vv',
    'efc__D',
    'ne_ten',
    'contact__solimp',
    'contact__dim',
    'collision_pair',
    'contact__geom',
    'ncollision',
    'efc__pos',
    'efc__margin',
    'efc__aref',
    'efc__uv',
    'efc__worldid',
}


def _to_elt(cont, _, d, axis):
  return DataWarp(**{
      f.name: (
          cont(getattr(d, f.name), axis)
          if f.name not in _DATA_NON_VMAP
          else getattr(d, f.name)
      )
      for f in DataWarp.fields()
  })


def _from_elt(cont, axis_size, d, axis_dest):
  return DataWarp(**{
      f.name: (
          cont(axis_size, getattr(d, f.name), axis_dest)
          if f.name not in _DATA_NON_VMAP
          else getattr(d, f.name)
      )
      for f in DataWarp.fields()
  })


batching.register_vmappable(DataWarp, int, int, _to_elt, _from_elt, None)

NDIM_TYPE = {
    'Model': {
        'nq': 0,
        'nv': 0,
        'nu': 0,
        'na': 0,
        'nbody': 0,
        'njnt': 0,
        'ngeom': 0,
        'nsite': 0,
        'ncam': 0,
        'nlight': 0,
        'nflex': 0,
        'nflexvert': 0,
        'nflexedge': 0,
        'nflexelem': 0,
        'nflexelemdata': 0,
        'nexclude': 0,
        'neq': 0,
        'nmocap': 0,
        'ngravcomp': 0,
        'nM': 0,
        'ntendon': 0,
        'nwrap': 0,
        'nsensor': 0,
        'nsensordata': 0,
        'nmeshvert': 0,
        'nmeshface': 0,
        'nlsp': 0,
        'npair': 0,
        'opt__timestep': 0,
        'opt__impratio': 0,
        'opt__tolerance': 0,
        'opt__ls_tolerance': 0,
        'opt__gravity': 2,
        'opt__integrator': 0,
        'opt__cone': 0,
        'opt__solver': 0,
        'opt__iterations': 0,
        'opt__ls_iterations': 0,
        'opt__disableflags': 0,
        'opt__is_sparse': 0,
        'opt__gjk_iterations': 0,
        'opt__epa_iterations': 0,
        'opt__epa_exact_neg_distance': 0,
        'opt__depth_extension': 0,
        'opt__ls_parallel': 0,
        'opt__wind': 2,
        'opt__density': 0,
        'opt__viscosity': 0,
        'stat__meaninertia': 0,
        'qpos0': 2,
        'qpos_spring': 2,
        'qM_fullm_i': 1,
        'qM_fullm_j': 1,
        'qM_mulm_i': 1,
        'qM_mulm_j': 1,
        'qM_madr_ij': 1,
        'qLD_updates': 0,
        'M_rownnz': 1,
        'M_rowadr': 1,
        'M_colind': 1,
        'mapM2M': 1,
        'qM_tiles': 0,
        'body_tree': 0,
        'body_parentid': 1,
        'body_rootid': 1,
        'body_weldid': 1,
        'body_mocapid': 1,
        'body_jntnum': 1,
        'body_jntadr': 1,
        'body_dofnum': 1,
        'body_dofadr': 1,
        'body_geomnum': 1,
        'body_geomadr': 1,
        'body_pos': 3,
        'body_quat': 3,
        'body_ipos': 3,
        'body_iquat': 3,
        'body_mass': 2,
        'body_subtreemass': 2,
        'subtree_mass': 2,
        'body_inertia': 3,
        'body_invweight0': 3,
        'body_contype': 1,
        'body_conaffinity': 1,
        'body_gravcomp': 2,
        'jnt_type': 1,
        'jnt_qposadr': 1,
        'jnt_dofadr': 1,
        'jnt_bodyid': 1,
        'jnt_limited': 1,
        'jnt_actfrclimited': 1,
        'jnt_solref': 3,
        'jnt_solimp': 3,
        'jnt_pos': 3,
        'jnt_axis': 3,
        'jnt_stiffness': 2,
        'jnt_range': 3,
        'jnt_actfrcrange': 3,
        'jnt_margin': 2,
        'jnt_limited_slide_hinge_adr': 1,
        'jnt_limited_ball_adr': 1,
        'jnt_actgravcomp': 1,
        'dof_bodyid': 1,
        'dof_jntid': 1,
        'dof_parentid': 1,
        'dof_Madr': 1,
        'dof_armature': 2,
        'dof_damping': 2,
        'dof_invweight0': 2,
        'dof_frictionloss': 2,
        'dof_solimp': 3,
        'dof_solref': 3,
        'dof_tri_row': 1,
        'dof_tri_col': 1,
        'geom_type': 1,
        'geom_contype': 1,
        'geom_conaffinity': 1,
        'geom_condim': 1,
        'geom_bodyid': 1,
        'geom_dataid': 1,
        'geom_group': 1,
        'geom_matid': 2,
        'geom_priority': 1,
        'geom_solmix': 2,
        'geom_solref': 3,
        'geom_solimp': 3,
        'geom_size': 3,
        'geom_aabb': 2,
        'geom_rbound': 2,
        'geom_pos': 3,
        'geom_quat': 3,
        'geom_friction': 3,
        'geom_margin': 2,
        'geom_gap': 2,
        'geom_rgba': 3,
        'site_type': 1,
        'site_bodyid': 1,
        'site_size': 2,
        'site_pos': 3,
        'site_quat': 3,
        'cam_mode': 1,
        'cam_bodyid': 1,
        'cam_targetbodyid': 1,
        'cam_pos': 3,
        'cam_quat': 3,
        'cam_poscom0': 3,
        'cam_pos0': 3,
        'cam_mat0': 4,
        'cam_fovy': 1,
        'cam_resolution': 2,
        'cam_sensorsize': 2,
        'cam_intrinsic': 2,
        'light_mode': 1,
        'light_bodyid': 1,
        'light_targetbodyid': 1,
        'light_pos': 3,
        'light_dir': 3,
        'light_poscom0': 3,
        'light_pos0': 3,
        'light_dir0': 3,
        'flex_dim': 1,
        'flex_vertadr': 1,
        'flex_vertnum': 1,
        'flex_edgeadr': 1,
        'flex_elemedgeadr': 1,
        'flex_vertbodyid': 1,
        'flex_edge': 2,
        'flex_elem': 1,
        'flex_elemedge': 1,
        'flexedge_length0': 1,
        'flex_stiffness': 1,
        'flex_damping': 1,
        'mesh_vertadr': 1,
        'mesh_vertnum': 1,
        'mesh_vert': 2,
        'mesh_faceadr': 1,
        'mesh_face': 2,
        'eq_type': 1,
        'eq_obj1id': 1,
        'eq_obj2id': 1,
        'eq_objtype': 1,
        'eq_active0': 1,
        'eq_solref': 3,
        'eq_solimp': 3,
        'eq_data': 3,
        'eq_connect_adr': 1,
        'eq_wld_adr': 1,
        'eq_jnt_adr': 1,
        'eq_ten_adr': 1,
        'actuator_moment_tiles_nv': 0,
        'actuator_moment_tiles_nu': 0,
        'actuator_affine_bias_gain': 0,
        'actuator_trntype': 1,
        'actuator_dyntype': 1,
        'actuator_gaintype': 1,
        'actuator_biastype': 1,
        'actuator_trnid': 2,
        'actuator_actadr': 1,
        'actuator_actnum': 1,
        'actuator_ctrllimited': 1,
        'actuator_forcelimited': 1,
        'actuator_actlimited': 1,
        'actuator_dynprm': 3,
        'actuator_gainprm': 3,
        'actuator_biasprm': 3,
        'actuator_ctrlrange': 3,
        'actuator_forcerange': 3,
        'actuator_actrange': 3,
        'actuator_gear': 3,
        'nxn_geom_pair': 2,
        'nxn_pairid': 1,
        'pair_dim': 1,
        'pair_geom1': 1,
        'pair_geom2': 1,
        'pair_solref': 3,
        'pair_solreffriction': 3,
        'pair_solimp': 3,
        'pair_margin': 2,
        'pair_gap': 2,
        'pair_friction': 3,
        'exclude_signature': 1,
        'condim_max': 0,
        'tendon_adr': 1,
        'tendon_num': 1,
        'tendon_limited': 1,
        'tendon_limited_adr': 1,
        'tendon_solref_lim': 3,
        'tendon_solimp_lim': 3,
        'tendon_solref_fri': 3,
        'tendon_solimp_fri': 3,
        'tendon_range': 3,
        'tendon_margin': 2,
        'tendon_stiffness': 2,
        'tendon_damping': 2,
        'tendon_frictionloss': 2,
        'tendon_lengthspring': 3,
        'tendon_length0': 2,
        'tendon_invweight0': 2,
        'wrap_objid': 1,
        'wrap_prm': 1,
        'wrap_type': 1,
        'tendon_jnt_adr': 1,
        'tendon_site_adr': 1,
        'tendon_site_pair_adr': 1,
        'ten_wrapadr_site': 1,
        'ten_wrapnum_site': 1,
        'wrap_jnt_adr': 1,
        'wrap_site_adr': 1,
        'wrap_site_pair_adr': 1,
        'sensor_type': 1,
        'sensor_datatype': 1,
        'sensor_objtype': 1,
        'sensor_objid': 1,
        'sensor_reftype': 1,
        'sensor_refid': 1,
        'sensor_dim': 1,
        'sensor_adr': 1,
        'sensor_cutoff': 1,
        'sensor_pos_adr': 1,
        'sensor_vel_adr': 1,
        'sensor_acc_adr': 1,
        'sensor_touch_adr': 1,
        'sensor_subtree_vel': 0,
        'sensor_rne_postconstraint': 0,
        'mocap_bodyid': 1,
        'mat_rgba': 3,
    },
    'Data': {
        'nworld': 0,
        'nconmax': 0,
        'njmax': 0,
        'ncon': 1,
        'ne': 1,
        'ne_connect': 1,
        'ne_weld': 1,
        'ne_jnt': 1,
        'ne_ten': 1,
        'nf': 1,
        'nl': 1,
        'nefc': 1,
        'time': 1,
        'energy': 2,
        'qpos': 2,
        'qvel': 2,
        'act': 2,
        'qacc_warmstart': 2,
        'ctrl': 2,
        'qfrc_applied': 2,
        'xfrc_applied': 3,
        'fluid_applied': 3,
        'eq_active': 2,
        'mocap_pos': 3,
        'mocap_quat': 3,
        'qacc': 2,
        'act_dot': 2,
        'xpos': 3,
        'xquat': 3,
        'xmat': 4,
        'xipos': 3,
        'ximat': 4,
        'xanchor': 3,
        'xaxis': 3,
        'geom_xpos': 3,
        'geom_xmat': 4,
        'site_xpos': 3,
        'site_xmat': 4,
        'cam_xpos': 3,
        'cam_xmat': 4,
        'light_xpos': 3,
        'light_xdir': 3,
        'subtree_com': 3,
        'cdof': 3,
        'cinert': 3,
        'flexvert_xpos': 3,
        'flexedge_length': 2,
        'flexedge_velocity': 2,
        'actuator_length': 2,
        'actuator_moment': 3,
        'crb': 3,
        'qM': 3,
        'qLD': 3,
        'qLDiagInv': 2,
        'ten_velocity': 2,
        'actuator_velocity': 2,
        'cvel': 3,
        'cdof_dot': 3,
        'qfrc_bias': 2,
        'qfrc_spring': 2,
        'qfrc_damper': 2,
        'qfrc_gravcomp': 2,
        'qfrc_fluid': 2,
        'qfrc_passive': 2,
        'subtree_linvel': 3,
        'subtree_angmom': 3,
        'subtree_bodyvel': 3,
        'actuator_force': 2,
        'qfrc_actuator': 2,
        'qfrc_smooth': 2,
        'qacc_smooth': 2,
        'qfrc_constraint': 2,
        'contact__dist': 1,
        'contact__pos': 2,
        'contact__frame': 3,
        'contact__includemargin': 1,
        'contact__friction': 2,
        'contact__solref': 2,
        'contact__solreffriction': 2,
        'contact__solimp': 2,
        'contact__dim': 1,
        'contact__geom': 2,
        'contact__efc_address': 2,
        'contact__worldid': 1,
        'efc__worldid': 1,
        'efc__id': 1,
        'efc__J': 2,
        'efc__pos': 1,
        'efc__margin': 1,
        'efc__D': 1,
        'efc__aref': 1,
        'efc__frictionloss': 1,
        'efc__force': 1,
        'efc__Jaref': 1,
        'efc__Ma': 2,
        'efc__grad': 2,
        'efc__grad_dot': 1,
        'efc__Mgrad': 2,
        'efc__search': 2,
        'efc__search_dot': 1,
        'efc__gauss': 1,
        'efc__cost': 1,
        'efc__prev_cost': 1,
        'efc__solver_niter': 1,
        'efc__active': 1,
        'efc__gtol': 1,
        'efc__mv': 2,
        'efc__jv': 1,
        'efc__quad': 2,
        'efc__quad_gauss': 2,
        'efc__h': 3,
        'efc__alpha': 1,
        'efc__prev_grad': 2,
        'efc__prev_Mgrad': 2,
        'efc__beta': 1,
        'efc__beta_num': 1,
        'efc__beta_den': 1,
        'efc__done': 1,
        'efc__ls_done': 1,
        'efc__p0': 2,
        'efc__lo': 2,
        'efc__lo_alpha': 1,
        'efc__hi': 2,
        'efc__hi_alpha': 1,
        'efc__lo_next': 2,
        'efc__lo_next_alpha': 1,
        'efc__hi_next': 2,
        'efc__hi_next_alpha': 1,
        'efc__mid': 2,
        'efc__mid_alpha': 1,
        'efc__cost_candidate': 2,
        'efc__quad_total_candidate': 3,
        'efc__u': 2,
        'efc__uu': 1,
        'efc__uv': 1,
        'efc__vv': 1,
        'efc__condim': 1,
        'qpos_t0': 2,
        'qvel_t0': 2,
        'act_t0': 2,
        'qvel_rk': 2,
        'qacc_rk': 2,
        'act_dot_rk': 2,
        'qfrc_integration': 2,
        'qacc_integration': 2,
        'act_vel_integration': 2,
        'qM_integration': 3,
        'qLD_integration': 3,
        'qLDiagInv_integration': 2,
        'sap_projection_lower': 2,
        'sap_projection_upper': 2,
        'sap_sort_index': 2,
        'sap_range': 2,
        'sap_cumulative_sum': 1,
        'sap_segment_index': 1,
        'collision_pair': 2,
        'collision_pairid': 1,
        'collision_worldid': 1,
        'ncollision': 1,
        'cacc': 3,
        'cfrc_int': 3,
        'cfrc_ext': 3,
        'ten_length': 2,
        'ten_J': 3,
        'ten_wrapadr': 2,
        'ten_wrapnum': 2,
        'wrap_obj': 3,
        'wrap_xpos': 3,
        'sensordata': 2,
    },
    'Option': {
        'timestep': 0,
        'impratio': 0,
        'tolerance': 0,
        'ls_tolerance': 0,
        'gravity': 2,
        'integrator': 0,
        'cone': 0,
        'solver': 0,
        'iterations': 0,
        'ls_iterations': 0,
        'disableflags': 0,
        'is_sparse': 0,
        'gjk_iterations': 0,
        'epa_iterations': 0,
        'epa_exact_neg_distance': 0,
        'depth_extension': 0,
        'ls_parallel': 0,
        'wind': 2,
        'density': 0,
        'viscosity': 0,
    },
    'Statistic': {'meaninertia': 0},
}
