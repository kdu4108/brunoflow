from colorama import Fore, Style
from enum import Enum
import itertools
import numpy as np
from typing import Union
from brunoflow.ad import name, Node


class SemiringEdgeType(Enum):
    """Enum representing the different kinds of weights associated with an edge in the computation graph."""

    ENTROPY = 1
    ABS_VAL_GRAD = 2
    GRAD = 3


# Map from the SemiringEdgeType -> the name of the key in the `out_grad` dict
# corresponding to the accumulated value for that semiring.
SEMIRING_EDGE_TYPE_TO_OUT_GRAD_DICT_KEY = {
    SemiringEdgeType.ENTROPY: "out_entropy",
    SemiringEdgeType.ABS_VAL_GRAD: "out_abs_val_grad",
}

# Name of the possible key-sets of the `out_grad` dict from which to infer the SemiringEdgeType
ENTROPY_SEMIRING_OUT_GRAD_KEYS = set(["out_entropy", "out_abs_val_grad"])
ABS_VAL_GRAD_OUT_GRAD_KEYS = set(["out_abs_val_grad"])


def get_semiring_edge_type(out_grad: Union[dict, int]) -> SemiringEdgeType:
    """Given a value for out_grad, infer and return what the SemiringEdgeType is."""
    if isinstance(out_grad, dict):
        out_grad_keys = set(out_grad.keys())
        if out_grad_keys == ENTROPY_SEMIRING_OUT_GRAD_KEYS:
            return SemiringEdgeType.ENTROPY
        elif out_grad_keys == ABS_VAL_GRAD_OUT_GRAD_KEYS:
            return SemiringEdgeType.ABS_VAL_GRAD
        else:
            raise ValueError(
                f"The `out_grad` parameter to the backward pass of this function is a dict that does not exactly match the keys for `entropy` or `abs_val_grad`.\nout_grad value: {out_grad}.\nMake sure you only pass in the necessary keys for your semiring!"
            )
    else:
        return SemiringEdgeType.GRAD


def get_relevant_out_grad_val(out_grad: Union[dict, np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Extracts the relevant property from the value passed to out_grad.

    :param out_grad: Either a float/np.ndarray or a dict representing the upstream out_grad value of a computation graph, where out_grad could refer to any accumulated value from running backprop on a semiring.
        If out_grad is a float/np.ndarray, then you can just return it because it's already either
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
