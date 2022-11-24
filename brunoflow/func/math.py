"""
This module contains definitions for autodiff versions of functions in the 'math' module
"""

from colorama import Fore, Style
import numpy as np
from jax import numpy as jnp
from .function import make_function, pointwise_backward
from brunoflow.func.utils import construct_single_variable_fct_name, construct_double_variable_fct_name
from ..ad import Node, name


##################### EXPONENTS #####################

sqrt = make_function(
    lambda x: np.sqrt(x),
    pointwise_backward(lambda out, x: 1 / (2 * out)),
    construct_single_variable_fct_name("sqrt"),
)

exp = make_function(
    lambda x: jnp.exp(x), pointwise_backward(lambda out, x: out), construct_single_variable_fct_name("exp")
)

log = make_function(
    lambda x: np.log(x), pointwise_backward(lambda out, x: 1 / x), construct_single_variable_fct_name("log")
)

##################### TRIGONOMETRIC #####################

sin = make_function(
    lambda x: np.sin(x),
    pointwise_backward(lambda out, x: np.cos(x)),
    construct_single_variable_fct_name("sin"),
)

cos = make_function(
    lambda x: np.cos(x), pointwise_backward(lambda out, x: -np.sin(x)), construct_single_variable_fct_name("cos")
)

tan = make_function(
    lambda x: np.tan(x), pointwise_backward(lambda out, x: 1 + out * out), construct_single_variable_fct_name("tan")
)

asin = make_function(
    lambda x: np.arcsin(x),
    pointwise_backward(lambda out, x: 1 / np.sqrt(1 - x * x)),
    construct_single_variable_fct_name("asin"),
)

acos = make_function(
    lambda x: np.arccos(x),
    pointwise_backward(lambda out, x: -1 / np.sqrt(1 - x * x)),
    construct_single_variable_fct_name("acos"),
)

atan = make_function(
    lambda x: np.arctan(x),
    pointwise_backward(lambda out, x: 1 / (1 + x * x)),
    construct_single_variable_fct_name("atan"),
)

atan2 = make_function(
    lambda a, b: np.arctan2(a, b),
    pointwise_backward(lambda out, a, b: (b / (a * a + b * b), -a / (a * a + b * b))),
    construct_double_variable_fct_name("atan2"),
)
Node.__np2bf[np.arctan2] = atan2

##################### HYPERBOLIC #####################

sinh = make_function(
    lambda x: np.sinh(x), pointwise_backward(lambda out, x: np.cosh(x)), construct_single_variable_fct_name("sinh")
)

cosh = make_function(
    lambda x: np.cosh(x), pointwise_backward(lambda out, x: np.sinh(x)), construct_single_variable_fct_name("cosh")
)

tanh = make_function(
    lambda x: np.tanh(x), pointwise_backward(lambda out, x: 1 - out * out), construct_single_variable_fct_name("tanh")
)

asinh = make_function(
    lambda x: np.arcsinh(x),
    pointwise_backward(lambda out, x: 1 / np.sqrt(x * x + 1)),
    construct_single_variable_fct_name("asinh"),
)

acosh = make_function(
    lambda x: np.arccosh(x),
    pointwise_backward(lambda out, x: 1 / np.sqrt(x * x - 1)),
    construct_single_variable_fct_name("acosh"),
)

atanh = make_function(
    lambda x: np.arctanh(x),
    pointwise_backward(lambda out, x: 1 / (1 - x * x)),
    construct_single_variable_fct_name("atanh"),
)

##################### EXTREMA #####################


def min_deriv(out, a, b):
    mask = (a < b).astype(float)
    return mask, 1 - mask


minimum = make_function(
    lambda a, b: np.minimum(a, b), pointwise_backward(min_deriv), construct_double_variable_fct_name("min")
)
Node.__np2bf[np.minimum] = minimum


def max_deriv(out, a, b):
    mask = (a > b).astype(float)
    return mask, 1 - mask


maximum = make_function(
    lambda a, b: np.maximum(a, b), pointwise_backward(max_deriv), construct_double_variable_fct_name("max")
)
Node.__np2bf[np.maximum] = maximum

##################### INTEGER CONVERSION #####################

floor = make_function(lambda x: np.floor(x), name_fct=construct_single_variable_fct_name("floor"))

ceil = make_function(lambda x: np.ceil(x), name_fct=construct_single_variable_fct_name("ceil"))

round = make_function(lambda x: np.round(x), name_fct=construct_single_variable_fct_name("round"))

##################### SANITY CHECKS #####################

isfinite = make_function(lambda x: np.isfinite(x), name_fct=construct_single_variable_fct_name("isfinite"))

isinf = make_function(lambda x: np.isinf(x), name_fct=construct_single_variable_fct_name("isinf"))

isnan = make_function(lambda x: np.isnan(x), name_fct=construct_single_variable_fct_name("isnan"))
