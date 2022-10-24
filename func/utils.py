from typing import Union
from colorama import Fore, Style
import itertools
from brunoflow.ad import name


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
USE_COLOR = True


def compute_additional_args_str(additional_arg_names, args):
    if len(additional_arg_names) != len(args):
        raise ValueError(
            f"len(additional_arg_name_list) is {len(additional_arg_names)}, which does not match len(args) which is {len(args)}"
        )
    out_str = " ".join([f"{additional_arg_names[i]}={args[i]}" for i in range(len(additional_arg_names))])
    if out_str:
        out_str = " " + out_str
    return out_str


def colorify(s):
    curr_color = COLOR_STACK.pop()
    return f"{curr_color[0]}{s}{curr_color[1]}"


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


ENTROPY_SEMIRING_OUT_GRAD_KEYS = set(["out_entropy", "out_abs_val_grad"])
ABS_VAL_GRAD_OUT_GRAD_KEYS = set(["out_abs_val_grad"])


def get_relevant_out_grad_val(out_grad: Union[dict, int]):
    """
    Extracts the relevant property from the value passed to out_grad.

    :param out_grad: Either an int or a dict representing the upstream out_grad value of a computation graph, where out_grad could refer to any accumulated value from running backprop on a semiring.
        If out_grad is an int, then you can just return it because it's already either
        1) the grad itself,
        2) the max pos grad, or
        3) the max neg grad.
    If out_grad is a dict, then you extract the appropriate value based on the keys in the dict.

    :return: the appropriate accumulated value of the semiring.
    """
    if isinstance(out_grad, dict):
        out_grad_keys = set(out_grad.keys())
        if out_grad_keys == ENTROPY_SEMIRING_OUT_GRAD_KEYS:
            out_grad = out_grad["out_entropy"]
        elif out_grad_keys == ABS_VAL_GRAD_OUT_GRAD_KEYS:
            out_grad = out_grad["out_abs_val_grad"]
        else:
            raise ValueError(
                f"The `out_grad` parameter to the backward pass of this function is a dict that does not exactly match the keys for `entropy` or `abs_val_grad`.\nout_grad value: {out_grad}.\nMake sure you only pass in the necessary keys for your semiring!"
            )

    return out_grad


# Used for debugging and seeing what values are there.
def print_vals_for_all_children(node, visited=set()):
    print(f"{name(node)}: {node.grad}, {node.abs_val_grad}, {node.entropy_wrt_output}")
    for inp in node.inputs:
        if isinstance(inp, Node) and inp not in visited:
            visited.add(inp)
            print_vals_for_all_children(inp, visited)
