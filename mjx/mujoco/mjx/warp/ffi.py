"""FFI helper functions for MJX."""

import inspect
import typing
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import warp as wp
from warp.jax_experimental import ffi
import functools
from jax import numpy as jp
import numpy as np
from mujoco.mjx.warp import types as mjx_warp_types


def flatten_tuple_signature(signature: inspect.Signature, args: Tuple):
  # https://gist.github.com/jaro-sevcik/cc90b939ecca86bee6f81e0bf560cc76
  def expand_parameter(parameter, arg_iter):
    if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
      try:
        arg = next(arg_iter)
        # If it is a tuple, we need to duplicate the parameter for each element.
        if isinstance(arg, tuple):
          assert typing.get_origin(p.annotation) == tuple, p.annotation
          type_args = typing.get_args(p.annotation)
          if len(type_args) == 2 and type_args[1] == ...:
            type_ = type_args[0]
            types = [type_]
            if hasattr(type_, '__dataclass_fields__'):
              # If the tuple element is a dataclass, we need to recycle the types
              # in order of the fields.
              fields = list(type_.__dataclass_fields__.values())
              types = [f.type for f in fields]
            return [
                inspect.Parameter(
                    f"{parameter.name}__{i}",
                    parameter.kind,
                    default=parameter.default,
                    annotation=types[i % len(types)],
                )
                for i in range(len(arg) * len(types))
            ]
          else:
            raise NotImplementedError(
                f"Unsupported tuple argument: {type_args} "
                "(currently, only Tuple[t, ...] is supported)."
            )
      except StopIteration:
        # We ran out input arguments.
        # Let us keep output parameters as is.
        pass
      assert typing.get_origin(p.annotation) != tuple, p.annotation
      return [parameter]
    else:
      raise ValueError(f"Unsupported parameter kind: {parameter.kind}")

  parameters = []
  arg_iter = iter(args)
  for p in signature.parameters.values():
    parameters.extend(expand_parameter(p, arg_iter))

  return inspect.Signature(
      parameters=parameters, return_annotation=signature.return_annotation
  )


def jax_callable_variadic_tuple(
    func: Callable,  # pylint: disable=g-bare-generic
    num_outputs: int = 1,
    graph_compatible: bool = True,
    vmap_method: Optional[str] = None,
    output_dims: dict[str, tuple[int, ...]]=None,
    in_out_argnames: Sequence[str]=None,
):
  """Wraps a JAX callable to flatten/unflatten variadic tuples."""
  def callable_wrapper(*args, **kwargs):
    def func_wrapper(*flat_args, **kwargs):
      unflat_args = jax.tree.unflatten(in_tree, flat_args)
      return func(*unflat_args, **kwargs)

    # Provide a flattened signature for the Warp callable machinery.
    func_wrapper.__signature__ = flatten_tuple_signature(
        inspect.signature(func), args
    )
    my_callable = ffi.jax_callable(
        func_wrapper,
        num_outputs=num_outputs,
        graph_compatible=graph_compatible,
        vmap_method=vmap_method,
        output_dims=output_dims,
        in_out_argnames=in_out_argnames,
    )

    flat_args, in_tree = jax.tree.flatten(args)
    return my_callable(*flat_args, **kwargs)

  return callable_wrapper


def _format_arg(arg: Any, name: str, annotation: Any, verbose: bool):
  """Formats a single argument for warp."""
  typ_args = typing.get_args(annotation)
  annotation_origin = typing.get_origin(annotation)

  # Handle variadic tuples.
  if annotation_origin == tuple and len(typ_args) == 2 and typ_args[1] == ...:
    return tuple(
        _format_arg(arg[i], name + f"_{i}", typ_args[0], verbose)
        for i in range(len(arg))
    )

  if not isinstance(annotation, wp.types.array):
    if verbose:
      print(f"Skipping {name}: {arg}")
    return arg

  expected_ndim = annotation.ndim
  if arg.ndim != expected_ndim:
    raise AssertionError('Arg ndim {arg.ndim} does not matche expected ndim {expected_ndim}.')

  # Add stride 0 to first axis in case the underlying argument is batched.
  # NB: the outer marshalling effectively does an "expand_dims".
  if arg.shape[0] == 1:
    old_strides = arg.strides
    arg.strides = (0,) + arg.strides[1:]
    if verbose:
      print(
          f"Leading batch dim of 1, adding stride: {name} {old_strides} =>"
          f" {arg.strides}"
      )
    return arg

  if verbose:
    print(f"Did nothing: {name}: {arg.shape}")
  return arg


def format_args_for_warp(func, verbose=False):
  @functools.wraps(func)
  def wrapper(*args):
    args = list(args)
    annotations = func.__annotations__
    assert len(args) == len(annotations)
    for i, (name, annotation) in enumerate(annotations.items()):
      args[i] = _format_arg(args[i], name, annotation, verbose)
    return func(*args)
  return wrapper


def _get_ndim_from_tree_path(path, ndim_map) -> Optional[int]:
  if isinstance(path, tuple):
    assert all(isinstance(p, jax.tree_util.GetAttrKey) for p in path)
    attr = '__'.join(p.name for p in path if p.name != '_impl')
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


def marshal_jax_warp_callable(func):
  """Marshal fields into a MuJoCo Warp function."""
  @functools.wraps(func)
  def wrapper(m, d):
    # Expand dims for Warp implicit vmap before calling into the FFI wrapped function.
    m_expanded = jax.tree.map_with_path(
      lambda path, x: _expand_dim_from_path(path, x, mjx_warp_types.NDIM_TYPE['Model']), m)
    d_expanded = jax.tree.map_with_path(
      lambda path, x: _expand_dim_from_path(path, x, mjx_warp_types.NDIM_TYPE['Data']), d)
    d_expanded_result = func(m_expanded, d_expanded)
    d_result = jax.tree.map(lambda n, o: _squeeze_dim(n, o), d_expanded_result, d)
    return d_result
  return wrapper
    

def _flatten_batch_dim(path: jax.tree_util.KeyPath, leaf: Any, ndim_map: dict[str, tuple[int, Any]]):
  ndim = _get_ndim_from_tree_path(path, ndim_map)
  if ndim is None:
    return leaf
  if ndim < leaf.ndim:
    assert leaf.ndim - ndim == 1
    batch_dim = np.prod(leaf.shape[:leaf.ndim - ndim + 1])
    return jp.reshape(leaf, (batch_dim,) + leaf.shape[leaf.ndim - ndim + 1:])
  return leaf


def _unflatten_batch_dim(leaf_squeezed: Any, leaf: Any) -> Any:
  if leaf_squeezed.ndim > leaf.ndim:
    raise AssertionError('Squeezed leaf ndim {leaf_squeezed.ndim} is greater than original leaf ndim {leaf.ndim}')
  if leaf_squeezed.ndim < leaf.ndim:
    return leaf_squeezed.reshape(leaf.shape)
  return leaf_squeezed


def marshal_custom_vmap(vmap_func):
  """Marshal fields for a custom vmap into an MuJoCo Warp function."""
  @functools.wraps(vmap_func)
  def wrapper(axis_size, is_batched, m, d):
    # Flatten batch dims into the first axis.
    m_flat = jax.tree.map_with_path(
        lambda path, x: _flatten_batch_dim(path, x, mjx_warp_types.NDIM_TYPE['Model']), m
    )
    d_flat = jax.tree.map_with_path(
        lambda path, x: _flatten_batch_dim(path, x, mjx_warp_types.NDIM_TYPE['Data']), d
    )
    d_flat_result, out_batched = vmap_func(axis_size, is_batched, m_flat, d_flat)
    # Unflatten batch dimension.
    d_result = jax.tree.map(
        lambda x, y: _unflatten_batch_dim(x, y), d_flat_result, d
    )
    return d_result, out_batched
  return wrapper
