import dataclasses
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi_helper
from mujoco.mjx.warp import types as mjx_warp_types
import mujoco_warp as mjwarp
import warp as wp
import jax
from typing import Any, Optional
from jax import numpy as jp
import numpy as np

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


def _kinematics(
    # Model
    ngeom: int,
    nsite: int,
    nmocap: int,
    qpos0: wp.array2d(dtype=float),
    body_tree: tuple[wp.array(dtype=int), ...],
    body_parentid: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    geom_bodyid: wp.array(dtype=int),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    site_bodyid: wp.array(dtype=int),
    site_pos: wp.array2d(dtype=wp.vec3),
    site_quat: wp.array2d(dtype=wp.quat),
    mocap_bodyid: wp.array(dtype=int),
    # Data
    qpos: wp.array2d(dtype=float),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
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
):
  _m.stat = _s
  _m.opt = _o
  _d.efc = _e
  _d.contact = _c
  _m.body_ipos = body_ipos
  _m.body_iquat = body_iquat
  _m.body_jntadr = body_jntadr
  _m.body_jntnum = body_jntnum
  _m.body_parentid = body_parentid
  _m.body_pos = body_pos
  _m.body_quat = body_quat
  _m.body_tree = body_tree
  _m.geom_bodyid = geom_bodyid
  _m.geom_pos = geom_pos
  _m.geom_quat = geom_quat
  _m.jnt_axis = jnt_axis
  _m.jnt_pos = jnt_pos
  _m.jnt_qposadr = jnt_qposadr
  _m.jnt_type = jnt_type
  _m.mocap_bodyid = mocap_bodyid
  _m.ngeom = ngeom
  _m.nmocap = nmocap
  _m.nsite = nsite
  _m.qpos0 = qpos0
  _m.site_bodyid = site_bodyid
  _m.site_pos = site_pos
  _m.site_quat = site_quat
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.mocap_pos = mocap_pos
  _d.mocap_quat = mocap_quat
  _d.qpos = qpos
  _d.site_xmat = site_xmat
  _d.site_xpos = site_xpos
  _d.xanchor = xanchor
  _d.xaxis = xaxis
  _d.ximat = ximat
  _d.xipos = xipos
  _d.xmat = xmat
  _d.xpos = xpos
  _d.xquat = xquat
  _d.nworld = _d.qpos.shape[0]
  mjwarp.kinematics(_m, _d)


def _kinematics_shim(
    ngeom: int,
    nsite: int,
    nmocap: int,
    qpos0: wp.array2d(dtype=float),
    body_tree: tuple[wp.array(dtype=int), ...],
    body_parentid: wp.array(dtype=int),
    body_jntnum: wp.array(dtype=int),
    body_jntadr: wp.array(dtype=int),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    geom_bodyid: wp.array(dtype=int),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    site_bodyid: wp.array(dtype=int),
    site_pos: wp.array2d(dtype=wp.vec3),
    site_quat: wp.array2d(dtype=wp.quat),
    mocap_bodyid: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
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
):
  args = (
      ngeom,
      nsite,
      nmocap,
      qpos0,
      body_tree,
      body_parentid,
      body_jntnum,
      body_jntadr,
      body_pos,
      body_quat,
      body_ipos,
      body_iquat,
      jnt_type,
      jnt_qposadr,
      jnt_pos,
      jnt_axis,
      geom_bodyid,
      geom_pos,
      geom_quat,
      site_bodyid,
      site_pos,
      site_quat,
      mocap_bodyid,
      qpos,
      mocap_pos,
      mocap_quat,
      xpos,
      xquat,
      xmat,
      xipos,
      ximat,
      xanchor,
      xaxis,
      geom_xpos,
      geom_xmat,
      site_xpos,
      site_xmat,
  )
  names = (
      "ngeom",
      "nsite",
      "nmocap",
      "qpos0",
      "body_tree",
      "body_parentid",
      "body_jntnum",
      "body_jntadr",
      "body_pos",
      "body_quat",
      "body_ipos",
      "body_iquat",
      "jnt_type",
      "jnt_qposadr",
      "jnt_pos",
      "jnt_axis",
      "geom_bodyid",
      "geom_pos",
      "geom_quat",
      "site_bodyid",
      "site_pos",
      "site_quat",
      "mocap_bodyid",
      "qpos",
      "mocap_pos",
      "mocap_quat",
      "xpos",
      "xquat",
      "xmat",
      "xipos",
      "ximat",
      "xanchor",
      "xaxis",
      "geom_xpos",
      "geom_xmat",
      "site_xpos",
      "site_xmat",
  )
  args = ffi_helper.format_args_for_warp(*args, names=names, kernel=_kinematics)
  # TODO(btaba): add stride 0 to arrays of first dim 1 since we effectively implement expand_dims below
  _kinematics(*args)

def _kinematics_callable_impl(m: types.Model, d: types.Data):
  output_dims = {
      "xpos": d.xpos.shape,
      "xquat": d.xquat.shape,
      "xmat": d.xmat.shape,
      "xipos": d.xipos.shape,
      "ximat": d.ximat.shape,
      "xanchor": d.xanchor.shape,
      "xaxis": d.xaxis.shape,
      "geom_xpos": d.geom_xpos.shape,
      "geom_xmat": d.geom_xmat.shape,
      "site_xpos": d.site_xpos.shape,
      "site_xmat": d.site_xmat.shape,
  }

  jf = ffi_helper.jax_callable_variadic_tuple(
      _kinematics_shim,
      num_outputs=11,
      output_dims=output_dims,
      vmap_method=None,  # all the vmap logic is handled in the custom vmap impl below
      graph_compatible=True,
      in_out_argnames={'xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'xanchor', 'xaxis',
                       'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat'},
  )
  out = jf(
      m.ngeom,
      m.nsite,
      m.nmocap,
      m.qpos0,
      m._impl.body_tree,
      m.body_parentid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_pos,
      m.jnt_axis,
      m.geom_bodyid,
      m.geom_pos,
      m.geom_quat,
      m.site_bodyid,
      m.site_pos,
      m.site_quat,
      m._impl.mocap_bodyid,
      d.qpos,
      d.mocap_pos,
      d.mocap_quat,
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
  )
  d = d.tree_replace({
      "xpos": out[0],
      "xquat": out[1],
      "xmat": out[2],
      "xipos": out[3],
      "ximat": out[4],
      "xanchor": out[5],
      "xaxis": out[6],
      "geom_xpos": out[7],
      "geom_xmat": out[8],
      "site_xpos": out[9],
      "site_xmat": out[10],
  })
  return d


def _get_ndim_from_tree_path(path, ndim_map) -> Optional[int]:
  if isinstance(path, tuple):
    assert all(isinstance(p, jax.tree_util.GetAttrKey) for p in path)
    attr = '__'.join(p.name for p in path)
    return ndim_map.get(attr)
  raise NotImplementedError(f'Parsing for jax tree path {path} not implemented.')


def _expand_dim_from_path(path: jax.tree_util.KeyPath, leaf: Any, ndim_map: dict[str, tuple[int, Any]]):
  ndim = _get_ndim_from_tree_path(path, ndim_map)
  if ndim is None:
    return leaf
  if ndim > leaf.ndim:
    return jp.expand_dims(leaf, axis=np.arange(ndim - leaf.ndim))
  if ndim < leaf.ndim:
    raise AssertionError(f'Leaf node ndim ({leaf.ndim}) must not have ndim greater than expected ndim: ({ndim}), for path {path}.')
  return leaf


def _squeeze_dim(leaf_expanded: Any, leaf: Any) -> Any:
  if leaf_expanded.ndim < leaf.ndim:
    raise AssertionError('Expanded leaf ndim {leaf_expaned.ndim} is smaller than original leaf ndim {leaf.ndim}')
  if leaf_expanded.ndim > leaf.ndim:
    return jp.squeeze(leaf_expanded, np.arange(leaf_expanded.ndim - leaf.ndim))
  return leaf_expanded



@jax.custom_batching.custom_vmap
def kinematics(m: types.Model, d: types.Data):
  # Expand dims for Warp implicit vmap before calling into the FFI wrapped function.
  m_expanded = jax.tree.map_with_path(
    lambda path, x: _expand_dim_from_path(path, x, mjx_warp_types.NDIM_TYPE['Model']), m)
  d_expanded = jax.tree.map_with_path(
    lambda path, x: _expand_dim_from_path(path, x, mjx_warp_types.NDIM_TYPE['Data']), d)
  d_expanded = _kinematics_callable_impl(m_expanded, d_expanded)
  d = jax.tree.map(lambda n, o: _squeeze_dim(n, o), d_expanded, d)
  return d


def _unflatten_batch_dim(leaf_squeezed: Any, leaf: Any) -> Any:
  if leaf_squeezed.ndim > leaf.ndim:
    raise AssertionError('Squeezed leaf ndim {leaf_squeezed.ndim} is greater than original leaf ndim {leaf.ndim}')
  if leaf_squeezed.ndim < leaf.ndim:
    return leaf_squeezed.reshape(leaf.shape)
  return leaf_squeezed


def _flatten_batch_dim(path: jax.tree_util.KeyPath, leaf: Any, ndim_map: dict[str, tuple[int, Any]]):
  ndim = _get_ndim_from_tree_path(path, ndim_map)
  if ndim is None:
    return leaf
  if ndim < leaf.ndim:
    assert leaf.ndim - ndim == 1
    return jp.reshape(leaf, (-1,) + leaf.shape[leaf.ndim - ndim + 1:])
  return leaf



@kinematics.def_vmap
def kinematics_vmap(axis_size, is_batched, m, d):
  # Flatten batch dims into the first axis, as expected by MuJoCo Warp.
  m_flat = jax.tree.map_with_path(
    lambda path, x: _flatten_batch_dim(path, x, mjx_warp_types.NDIM_TYPE['Model']), m
  )
  d_flat = jax.tree.map_with_path(
    lambda path, x: _flatten_batch_dim(path, x, mjx_warp_types.NDIM_TYPE['Data']), d
  )
  d_flat = kinematics(m_flat, d_flat)
  d = jax.tree.map(lambda x, y: _unflatten_batch_dim(x, y), d_flat, d)
  return d, is_batched[1]
