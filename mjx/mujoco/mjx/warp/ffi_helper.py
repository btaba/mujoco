"""FFI helper functions for MJX."""

import logging
from typing import Any

import numpy as np
import warp as wp


def adr_arr_to_tuple(
    arr_val: wp.array(dtype=int), arr_adr: wp.array(dtype=int)
) -> tuple[wp.array, ...]:
  arr_list = []
  arr_adr = arr_adr.numpy()
  arr_val = arr_val.numpy()
  for i in range(len(arr_adr) - 1):
    beg = arr_adr[i]
    end = arr_adr[i + 1]
    arr_list.append(wp.array(arr_val[beg:end]))
  return tuple(arr_list)


def tuple_to_adr_arr(arr: tuple[Any, ...]) -> tuple[Any, Any]:
  arr_val = np.concatenate(arr, dtype=np.int32)
  arr_adr = np.cumsum([len(v) for v in arr], dtype=np.int32)
  arr_adr = np.append(np.array(0), arr_adr).astype(np.int32)
  return arr_val, arr_adr


def format_args_for_warp(
    *args: Any, names: tuple[str, ...], kernel: Any
) -> Any:
  """Formats args for warp assuming vmap_method="expand_dims"."""
  new_args = [None] * len(args)
  annotations = tuple(kernel.__annotations__.items())
  for i in range(len(args)):
    if not hasattr(annotations[i][1], 'ndim'):
      new_args[i] = args[i]
      continue
    expected_ndim = annotations[i][1].ndim

    # Remove the expanded_dim if we are exceeding the expected ndim.
    # i.e. Unbatched model fields will get an extra dim due to
    # vmap_method="expand_dims".
    if args[i].ndim > expected_ndim and args[i].shape[0] == 1:
      extra_dim = args[i].ndim - expected_ndim
      assert sum(args[i].shape[:extra_dim]) == extra_dim
      arg = args[i].reshape(args[i].shape[extra_dim:])
      arg.ndim = expected_ndim
      new_args[i] = arg
      logging.debug(
          "Removing extra dim: %s %s => %s",
          names[i],
          args[i].shape,
          new_args[i].shape,  # pytype: disable=attribute-error
      )
      if new_args[i].ndim > 1 and new_args[i].shape[0] == 1:
        old_strides = new_args[i].strides
        new_args[i].strides = (0,) + new_args[i].strides[1:]
        new_args[i] = new_args[i]
        logging.debug(
            "Leading batch dim of 1, adding stride: %s %s => %s",
            names[i],
            old_strides,
            new_args[i].strides,  # pytype: disable=attribute-error
        )
      continue

    # Squash nested vmap batch axes.
    if args[i].ndim > expected_ndim:
      extra_ndim = args[i].ndim - expected_ndim
      new_args[i] = args[i].reshape((-1,) + args[i].shape[extra_ndim + 1 :])
      new_args[i].ndim = expected_ndim
      logging.debug(
          "Squashing extra dim: %s %s => %s",
          names[i],
          args[i].shape,
          new_args[i].shape,  # pytype: disable=attribute-error
      )
      continue

    # Add stride 0 to unbatched inputs that have the correct ndim.
    # Unbatched inputs get a leading dimension of 1, using
    # vmap_method="expand_dims". We add a stride of 0 to the leading dim.
    if expected_ndim == args[i].ndim and args[i].shape[0] == 1:
      arg = args[i]
      old_strides = arg.strides
      arg.strides = (0,) + arg.strides[1:]
      new_args[i] = arg
      logging.debug(
          "Leading batch dim of 1, adding stride: %s %s => %s",
          names[i],
          old_strides,
          new_args[i].strides,  # pytype: disable=attribute-error
      )
      continue

    # Add batch dims if they don't exist. This occurs when the underlying
    # function expects a batch dim but the outer function was not called
    # with vmap.
    # e.g. Model/Data have nworld == 1, but without the leading dim in JAX.
    if expected_ndim > args[i].ndim:
      extra_dims = expected_ndim - args[i].ndim
      new_args[i] = args[i].reshape((1,) * extra_dims + args[i].shape)
      new_args[i].ndim = expected_ndim
      logging.debug(
          "No leading batch dims %s %s => %s",
          names[i],
          args[i].shape,
          new_args[i].shape,  # pytype: disable=attribute-error
      )
      continue

    new_args[i] = args[i]
    logging.debug("Did nothing: %s %s => %s", names[i], args[i].shape, new_args[i].shape)  # pytype: disable=attribute-error

  return new_args
