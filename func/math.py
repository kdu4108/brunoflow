"""
This module contains definitions for autodiff versions of functions in the 'math' module
"""

import numpy as np
from .function import make_function, pointwise_backward
from ..ad import Node, name

##################### EXPONENTS #####################

sqrt = make_function(
    lambda x: np.sqrt(x), pointwise_backward(lambda out, x: 1 / (2 * out)), lambda x: f"(sqrt {name(x)})"
)

exp = make_function(lambda x: np.exp(x), pointwise_backward(lambda out, x: out), lambda x: f"(exp {name(x)})")

log = make_function(lambda x: np.log(x), pointwise_backward(lambda out, x: 1 / x), lambda x: f"(log {name(x)})")

##################### TRIGONOMETRIC #####################

sin = make_function(
    lambda x: np.sin(x),
    pointwise_backward(lambda out, x: np.cos(x)),
    lambda x: f"(sin {name(x)})",
)

cos = make_function(lambda x: np.cos(x), pointwise_backward(lambda out, x: -np.sin(x)), lambda x: f"(cos {name(x)})")

tan = make_function(lambda x: np.tan(x), pointwise_backward(lambda out, x: 1 + out * out), lambda x: f"(tan {name(x)})")

asin = make_function(
    lambda x: np.arcsin(x), pointwise_backward(lambda out, x: 1 / np.sqrt(1 - x * x)), lambda x: f"(asin {name(x)})"
)

acos = make_function(
    lambda x: np.arccos(x), pointwise_backward(lambda out, x: -1 / np.sqrt(1 - x * x)), lambda x: f"(acos {name(x)})"
)

atan = make_function(
    lambda x: np.arctan(x), pointwise_backward(lambda out, x: 1 / (1 + x * x)), lambda x: f"(atan {name(x)})"
)

atan2 = make_function(
    lambda a, b: np.arctan2(a, b),
    pointwise_backward(lambda out, a, b: (b / (a * a + b * b), -a / (a * a + b * b))),
    lambda a, b: f"(atan2 {name(a)} {name(b)})",
)
Node.__np2bf[np.arctan2] = atan2

##################### HYPERBOLIC #####################

sinh = make_function(lambda x: np.sinh(x), pointwise_backward(lambda out, x: np.cosh(x)), lambda x: f"(sinh {name(x)})")

cosh = make_function(lambda x: np.cosh(x), pointwise_backward(lambda out, x: np.sinh(x)), lambda x: f"(cosh {name(x)})")

tanh = make_function(
    lambda x: np.tanh(x), pointwise_backward(lambda out, x: 1 - out * out), lambda x: f"(tanh {name(x)})"
)

asinh = make_function(
    lambda x: np.arcsinh(x), pointwise_backward(lambda out, x: 1 / np.sqrt(x * x + 1)), lambda x: f"(asinh {name(x)})"
)

acosh = make_function(
    lambda x: np.arccosh(x), pointwise_backward(lambda out, x: 1 / np.sqrt(x * x - 1)), lambda x: f"(acosh {name(x)})"
)

atanh = make_function(
    lambda x: np.arctanh(x), pointwise_backward(lambda out, x: 1 / (1 - x * x)), lambda x: f"(atanh {name(x)})"
)

##################### EXTREMA #####################


def min_deriv(out, a, b):
    mask = (a < b).astype(float)
    return mask, 1 - mask


minimum = make_function(
    lambda a, b: np.minimum(a, b), pointwise_backward(min_deriv), lambda a, b: f"(min {name(a)} {name(b)})"
)
Node.__np2bf[np.minimum] = minimum


def max_deriv(out, a, b):
    mask = (a > b).astype(float)
    return mask, 1 - mask


maximum = make_function(
    lambda a, b: np.maximum(a, b),
    pointwise_backward(max_deriv),
    lambda a, b: f"(max {name(a)} {name(b)})",
)
Node.__np2bf[np.maximum] = maximum

##################### INTEGER CONVERSION #####################

floor = make_function(lambda x: np.floor(x), name_fct=lambda x: f"(floor {name(x)})")

ceil = make_function(lambda x: np.ceil(x), name_fct=lambda x: f"(ceil {name(x)})")

round = make_function(lambda x: np.round(x), name_fct=lambda x: f"(round {name(x)})")

##################### SANITY CHECKS #####################

isfinite = make_function(lambda x: np.isfinite(x), name_fct=lambda x: f"(isfinite {name(x)})")

isinf = make_function(lambda x: np.isinf(x), name_fct=lambda x: f"(isinf {name(x)})")

isnan = make_function(lambda x: np.isnan(x), name_fct=lambda x: f"(isnan {name(x)})")
