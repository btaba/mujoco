"""Warp-specific types."""

from typing import Tuple
import jax
from mujoco.mjx._src import dataclasses as mjx_dataclasses
import numpy as np
import dataclasses

PyTreeNode = mjx_dataclasses.PyTreeNode

from jax.tree_util import register_pytree_node_class

@dataclasses.dataclass(frozen=True)
@register_pytree_node_class
class TileSet:
  """Tiling configuration for decomposible block diagonal matrix."""
  adr: np.ndarray
  size: int

  # Provide custom flatten/unflatten logic for FFI to expand
  # fields for the JAX callable.
  def tree_flatten(self):
    children = (self.adr, self.size)
    return children, None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    adr_unflattened, size_unflattened = children
    return cls(adr=adr_unflattened, size=size_unflattened)


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
  mocap_bodyid: jax.Array
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
  ten_wrapadr_site: jax.Array
  ten_wrapnum_site: jax.Array
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
  collision_pair: np.ndarray
  collision_pairid: np.ndarray
  collision_worldid: np.ndarray
  contact__dim: np.ndarray
  contact__dist: np.ndarray
  contact__efc_address: np.ndarray
  contact__frame: np.ndarray
  contact__friction: np.ndarray
  contact__geom: np.ndarray
  contact__includemargin: np.ndarray
  contact__pos: np.ndarray
  contact__solimp: np.ndarray
  contact__solref: np.ndarray
  contact__solreffriction: np.ndarray
  contact__worldid: np.ndarray
  crb: jax.Array
  efc__D: jax.Array
  efc__J: np.ndarray
  efc__Jaref: np.ndarray
  efc__Ma: jax.Array
  efc__Mgrad: jax.Array
  efc__active: np.ndarray
  efc__alpha: jax.Array
  efc__aref: jax.Array
  efc__beta: jax.Array
  efc__beta_den: jax.Array
  efc__beta_num: jax.Array
  efc__condim: np.ndarray
  efc__cost: jax.Array
  efc__cost_candidate: jax.Array
  efc__done: jax.Array
  efc__force: np.ndarray
  efc__frictionloss: np.ndarray
  efc__gauss: jax.Array
  efc__grad: jax.Array
  efc__grad_dot: jax.Array
  efc__gtol: jax.Array
  efc__h: jax.Array
  efc__hi: jax.Array
  efc__hi_alpha: jax.Array
  efc__hi_next: jax.Array
  efc__hi_next_alpha: jax.Array
  efc__id: np.ndarray
  efc__jv: np.ndarray
  efc__lo: jax.Array
  efc__lo_alpha: jax.Array
  efc__lo_next: jax.Array
  efc__lo_next_alpha: jax.Array
  efc__ls_done: jax.Array
  efc__margin: np.ndarray
  efc__mid: jax.Array
  efc__mid_alpha: jax.Array
  efc__mv: jax.Array
  efc__p0: jax.Array
  efc__pos: np.ndarray
  efc__prev_Mgrad: jax.Array
  efc__prev_cost: jax.Array
  efc__prev_grad: jax.Array
  efc__quad: np.ndarray
  efc__quad_gauss: jax.Array
  efc__quad_total_candidate: jax.Array
  efc__search: jax.Array
  efc__search_dot: jax.Array
  efc__solver_niter: jax.Array
  efc__u: np.ndarray
  efc__uu: np.ndarray
  efc__uv: np.ndarray
  efc__vv: np.ndarray
  efc__worldid: np.ndarray
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
