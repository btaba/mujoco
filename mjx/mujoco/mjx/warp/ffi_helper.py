"""FFI helper functions for MJX."""

import inspect
import typing
from typing import Any, Callable, Sequence, Tuple

import jax
import warp as wp
from warp.jax_experimental.ffi import jax_callable
import functools


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
    *callable_args,
    **callable_kwargs,
):
  """Wraps a JAX callable to flatten/unflatten variadic tuples."""
  
  def callable_wrapper(*args, **kwargs):
    def func_wrapper(*flat_args, **kwargs):
      # num_inputs = len(flat_args) - num_outputs + len(in_out_argnames)
      # flat_inputs = flat_args[: num_inputs]
      # TODO(btaba): fix this...
      # self.output_args = [a for a in self.args if a.in_out] + self.args[self.num_inputs :]
      # outputs = flat_args[num_inputs:]
      # unflat_inputs = jax.tree.unflatten(in_tree, flat_inputs)
      # return func(*unflat_inputs + outputs, **kwargs)
      unflat_args = jax.tree.unflatten(in_tree, flat_args)
      return func(*unflat_args, **kwargs)

    # Provide a flattened signature for the Warp callable machinery.
    func_wrapper.__signature__ = flatten_tuple_signature(
        inspect.signature(func), args
    )
    my_callable = jax_callable(
        func_wrapper, num_outputs,
        *callable_args, **callable_kwargs
    )

    flat_args, in_tree = jax.tree.flatten(args)
    return my_callable(*flat_args, **kwargs)

  return callable_wrapper


def _format_arg(arg: Any, name: str, annotation: Any, verbose: bool):
  """Formats a single argument for warp."""
  # Handle variadic tuples.
  typ_args = typing.get_args(annotation)
  annotation_origin = typing.get_origin(annotation)
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

  # shape_flat = functools.reduce(lambda a, b: a * b, arg.shape)
  # if shape_flat == 0:
  #   if verbose:
  #     print(f"Skipping empty array {name}: {arg.shape}")
  #   arg = arg.reshape(())
  #   return arg

  # Remove the expanded_dim if we are exceeding the expected ndim.
  # i.e. Unbatched model fields will get an extra dim due to
  # vmap_method="expand_dims".
  if arg.ndim > expected_ndim and arg.shape[0] == 1:
    extra_dim = arg.ndim - expected_ndim
    assert sum(arg.shape[:extra_dim]) == extra_dim
    new_arg = arg.reshape(arg.shape[extra_dim:])
    new_arg.ndim = expected_ndim
    if verbose:
      print(f"Removing extra dim: {name} {arg.shape} => {new_arg.shape}")
    # The annotation has larger ndim but the leading dim is 1.
    # Let's add a stride of 0 to the first axis.
    if new_arg.ndim > 1 and new_arg.shape[0] == 1:
      old_strides = new_arg.strides
      new_arg.strides = (0,) + new_arg.strides[1:]
      if verbose:
        print(
            f"Leading batch dim of 1, adding stride: {name} {old_strides} =>"
            f" {new_arg.strides}"
        )
    return new_arg

  # Squash nested vmap batch axes.
  if arg.ndim > expected_ndim:
    extra_ndim = arg.ndim - expected_ndim
    # avoid -1 reshape if the size is 0
    new_arg = arg.reshape((functools.reduce(lambda a, b: a * b, arg.shape[:extra_ndim + 1]),) + arg.shape[extra_ndim + 1 :])
    new_arg.ndim = expected_ndim
    if verbose:
      print(f"Squashing extra dim: {name} {arg.shape} => {new_arg.shape}")
    return new_arg

  # Add stride 0 to unbatched inputs that have the correct ndim.
  # Unbatched inputs get a leading dimension of 1, using
  # vmap_method="expand_dims". We add a stride of 0 to the leading dim.
  if expected_ndim == arg.ndim and arg.shape[0] == 1:
    arg = arg
    old_strides = arg.strides
    arg.strides = (0,) + arg.strides[1:]
    new_arg = arg
    if verbose:
      print(
          f"Leading batch dim of 1, adding stride: {name} {old_strides} =>"
          f" {new_arg.strides}"
      )
    return new_arg

  # Add batch dims if they don't exist. This occurs when the underlying
  # function expects a batch dim but the outer function was not called
  # with vmap.
  # e.g. Model/Data have nworld == 1, but without the leading dim in JAX.
  if expected_ndim > arg.ndim:
    extra_dims = expected_ndim - arg.ndim
    new_arg = arg.reshape((1,) * extra_dims + arg.shape)
    new_arg.ndim = expected_ndim
    new_arg.strides = (0,) + new_arg.strides[1:]
    if verbose:
      print(f"No leading batch dims {name} {arg.shape} => {new_arg.shape}")
      print(f"Adding stride: {name} {arg.strides} => {new_arg.strides}")
    return new_arg

  if verbose:
    print(f"Did nothing: {name}: {arg.shape}")
  return arg


def format_args_for_warp(
    *args: Any, names: tuple[str, ...], kernel: Any, verbose: bool = True
) -> Any:
  """Formats args for warp assuming vmap_method="expand_dims"."""
  new_args = []
  annotations = kernel.__annotations__
  for i in range(len(args)):
    print(names[i])
    new_args.append(
        _format_arg(args[i], names[i], annotations[names[i]], verbose)
    )
  return new_args
