from colorama import Fore, Style
from enum import Enum
import itertools
from jax import numpy as jnp
from typing import Union
from brunoflow.ad import name, Node


class SemiringEdgeType(Enum):
    """Enum representing the different kinds of weights associated with an edge in the computation graph."""

    ENTROPY = 1
    ABS_VAL_GRAD = 2
    GRAD = 3
    MAX_POS_GRAD = 4
    MAX_NEG_GRAD = 5


# Map from the SemiringEdgeType -> the name of the key in the `out_grad` dict
# corresponding to the accumulated value for that semiring.
SEMIRING_EDGE_TYPE_TO_OUT_GRAD_DICT_KEY = {
    SemiringEdgeType.ENTROPY: "out_entropy",
    SemiringEdgeType.ABS_VAL_GRAD: "out_abs_val_grad",
    SemiringEdgeType.MAX_POS_GRAD: "out_max_pos_grad",
    SemiringEdgeType.MAX_NEG_GRAD: "out_max_neg_grad",
}

# Name of the possible key-sets of the `out_grad` dict from which to infer the SemiringEdgeType
ENTROPY_SEMIRING_OUT_GRAD_KEYS = set(["out_entropy", "out_abs_val_grad"])
ABS_VAL_GRAD_OUT_GRAD_KEYS = set(["out_abs_val_grad"])
MAX_POS_GRAD_OUT_GRAD_KEYS = set(["out_max_pos_grad"])
MAX_NEG_GRAD_OUT_GRAD_KEYS = set(["out_max_neg_grad"])


def get_semiring_edge_type(out_grad: Union[dict, int]) -> SemiringEdgeType:
    """Given a value for out_grad, infer and return what the SemiringEdgeType is."""
    if isinstance(out_grad, dict):
        out_grad_keys = set(out_grad.keys())
        if out_grad_keys == ENTROPY_SEMIRING_OUT_GRAD_KEYS:
            return SemiringEdgeType.ENTROPY
        elif out_grad_keys == ABS_VAL_GRAD_OUT_GRAD_KEYS:
            return SemiringEdgeType.ABS_VAL_GRAD
        elif out_grad_keys == MAX_POS_GRAD_OUT_GRAD_KEYS:
            return SemiringEdgeType.MAX_POS_GRAD
        elif out_grad_keys == MAX_NEG_GRAD_OUT_GRAD_KEYS:
            return SemiringEdgeType.MAX_NEG_GRAD
        else:
            raise ValueError(
                f"The `out_grad` parameter to the backward pass of this function is a dict that does not exactly match the keys for `entropy`, `abs_val_grad`, or `max_grad`.\nout_grad value: {out_grad}.\nMake sure you only pass in the necessary keys for your semiring!"
            )
    else:
        return SemiringEdgeType.GRAD


def get_relevant_out_grad_val(out_grad: Union[dict, jnp.ndarray, float]) -> Union[jnp.ndarray, float]:
    """
    Extracts the relevant property from the value passed to out_grad.

    :param out_grad: Either a float/jnp.ndarray or a dict representing the upstream out_grad value of a computation graph, where out_grad could refer to any accumulated value from running backprop on a semiring.
        If out_grad is a float/jnp.ndarray, then you can just return it because it's already either
        1) the grad itself,
        2) the max pos grad, or
        3) the max neg grad.
    If out_grad is a dict, then you extract the appropriate value based on the keys in the dict.

    :return: the appropriate accumulated value of the semiring.
    """
    semiring_edge_type = get_semiring_edge_type(out_grad)
    if semiring_edge_type in SEMIRING_EDGE_TYPE_TO_OUT_GRAD_DICT_KEY.keys():
        return out_grad[SEMIRING_EDGE_TYPE_TO_OUT_GRAD_DICT_KEY[semiring_edge_type]]
    return out_grad


COLORS = itertools.cycle(
    [
        Fore.BLUE,
        Fore.RED,
        Fore.GREEN,
        Fore.YELLOW,
        Fore.CYAN,
        Fore.MAGENTA,
        Fore.LIGHTBLUE_EX,
        Fore.LIGHTRED_EX,
        Fore.LIGHTGREEN_EX,
        Fore.LIGHTYELLOW_EX,
        Fore.LIGHTCYAN_EX,
        Fore.LIGHTMAGENTA_EX,
    ]
)
global COLOR_STACK
global color
COLOR_STACK = [(Style.RESET_ALL, Style.RESET_ALL)]
USE_COLOR = False


def colorify(s):
    curr_color = COLOR_STACK.pop()
    return f"{curr_color[0]}{s}{curr_color[1]}"


def compute_additional_args_str(additional_arg_names, args):
    if len(additional_arg_names) != len(args):
        raise ValueError(
            f"len(additional_arg_name_list) is {len(additional_arg_names)}, which does not match len(args) which is {len(args)}"
        )
    out_str = " ".join([f"{additional_arg_names[i]}={args[i]}" for i in range(len(additional_arg_names))])
    if out_str:
        out_str = " " + out_str
    return out_str


def construct_single_variable_fct_name(fct_name, additional_arg_names=tuple()):
    color = next(COLORS)
    COLOR_STACK.append((color, COLOR_STACK[-1][0]))

    def f(x, *args):
        output_str = f"({fct_name} {name(x)}{compute_additional_args_str(additional_arg_names, args)})"
        return colorify(output_str) if USE_COLOR else output_str

    return f


def construct_double_variable_fct_name(fct_name, additional_arg_names=tuple()):
    color = next(COLORS)
    COLOR_STACK.append((color, COLOR_STACK[-1][0]))

    def f(a, b, *args):
        output_str = f"({fct_name} {name(a)} {name(b)}{compute_additional_args_str(additional_arg_names, args)})"
        return colorify(output_str) if USE_COLOR else output_str

    return f


def print_vals_for_all_children(node, visited=set()):
    """
    Print properties (e.g. grad) of inputted node and all children node down to the leaves
    (where a node's inputs are its direct children). Good for debugging!
    """
    print(f"{name(node)}: {node.grad}, {node.abs_val_grad}, {node.entropy_wrt_output}")
    for inp in node.inputs:
        if isinstance(inp, Node) and inp not in visited:
            visited.add(inp)
            print_vals_for_all_children(inp, visited)


"""
Defining put_along_axis() from numpy for jax.numpy.
Essentially copied the code from
https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/lib/shape_base.py#L29

"""

import numpy.core.numeric as _nx
import jax


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError("`indices` must be an integer array")
    if len(arr_shape) != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    if axis == -1:
        axis = len(arr_shape) - 1

    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def custom_put_along_axis(arr, indices, values, axis):
    """
    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Destination array.
    indices : ndarray (Ni..., J, Nk...)
        Indices to change along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        against `arr`.
    values : array_like (Ni..., J, Nk...)
        values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is None, the destination
        array is treated as if a flattened 1d view had been created of it.

    """

    # normalize inputs
    if axis is None:
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter has no .shape
    else:
        # axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    arr = arr.at[tuple(_make_along_axis_idx(arr_shape, indices, axis))].set(values)
    return arr


def typecast_index_arg_for_jax(arg):
    """
    When indexing into a JAX array, if the indexing argument is a list, we need to cast it into an jnparray.
    Otherwise: `TypeError: Using a non-tuple sequence for multidimensional indexing is not allowed; use `arr[array(seq)]` instead of `arr[seq]`. See https://github.com/google/jax/issues/4564 for more information.`

    However, we can't do this for all args because if it's a tuple, we may be trying index a single item and receive a slice instead.
    """
    return jnp.array(arg) if isinstance(arg, list) else arg
