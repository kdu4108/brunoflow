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
USE_COLOR = False


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
