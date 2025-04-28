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
"""Warp-specific types."""

import jax
from mujoco.mjx._src import dataclasses

PyTreeNode = dataclasses.PyTreeNode


class OptionWarp(PyTreeNode):
  """Derived fields from Option."""

  depth_extension: float
  epa_exact_neg_distance: bool
  epa_iterations: int
  gjk_iterations: int
  is_sparse: bool
  ls_parallel: bool


class ModelWarp(PyTreeNode):
  """Derived fields from Model."""

  M_colind: jax.Array
  M_rowadr: jax.Array
  M_rownnz: jax.Array
  actuator_affine_bias_gain: bool
  actuator_moment_offset_nu: jax.Array
  actuator_moment_offset_nv: jax.Array
  actuator_moment_tileadr: jax.Array
  actuator_moment_tilesize_nu: jax.Array
  actuator_moment_tilesize_nv: jax.Array
  alpha_candidate: jax.Array
  body_tree: jax.Array
  body_treeadr: jax.Array
  condim_max: int
  dof_tri_col: jax.Array
  dof_tri_row: jax.Array
  eq_connect_adr: jax.Array
  eq_jnt_adr: jax.Array
  eq_wld_adr: jax.Array
  jnt_limited_ball_adr: jax.Array
  jnt_limited_slide_hinge_adr: jax.Array
  light_bodyid: jax.Array
  light_targetbodyid: jax.Array
  mapM2M: jax.Array
  nlsp: int
  nxn_geom_pair: jax.Array
  nxn_pairid: jax.Array
  qLD_tile: jax.Array
  qLD_tileadr: jax.Array
  qLD_tilesize: jax.Array
  qLD_update_tree: jax.Array
  qLD_update_treeadr: jax.Array
  qM_fullm_i: jax.Array
  qM_fullm_j: jax.Array
  qM_madr_ij: jax.Array
  qM_mulm_i: jax.Array
  qM_mulm_j: jax.Array
  sensor_acc_adr: jax.Array
  sensor_pos_adr: jax.Array
  sensor_vel_adr: jax.Array
  subtree_mass: jax.Array
  ten_wrapadr_site: jax.Array
  ten_wrapnum_site: jax.Array
  tendon_jnt_adr: jax.Array
  tendon_limited_adr: jax.Array
  tendon_site_adr: jax.Array
  tendon_site_pair_adr: jax.Array
  wrap_jnt_adr: jax.Array
  wrap_site_adr: jax.Array
  wrap_site_pair_adr: jax.Array


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
  light_xdir: jax.Array
  light_xpos: jax.Array
  ncollision: jax.Array
  ncon: jax.Array
  nconmax: int
  ne: jax.Array
  ne_connect: jax.Array
  ne_jnt: jax.Array
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
  rne_cacc: jax.Array
  rne_cfrc: jax.Array
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
