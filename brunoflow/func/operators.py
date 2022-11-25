"""
This module contains definitions for autodiff versions of functions that are implemented in
    Python as infix operators (e.g. +, -, *, /)

For unary operators (eg. unary -), the relevant overload method is implemented (e.g. __neg__)

For binary operators, two additional steps are taken:

(1) For an operator __[op]__, the 'reverse' operator __r[op]__ is implemented so that the
    correct function is called when the first argument to the operator is a scalar (int | float)
    and the second argument is a Node.

(2) An entry is added to Node.__np2bf so that the correct function is called when the first
    argument to the operator is a numpy.ndarray and the second argument is a Node.
"""

from jax import numpy as jnp
from .function import make_function, pointwise_backward
from ..ad import Node, name
from brunoflow.func.utils import construct_single_variable_fct_name, construct_double_variable_fct_name


def make_scalar_dispatch_op(op):
    def _op(a, b):
        if isinstance(b, float) or isinstance(b, int):
            return op(b, a)
        else:
            return NotImplemented

    return _op


##################### ARITHMETIC #####################

neg = make_function(lambda x: -x, pointwise_backward(lambda out, x: -1), construct_single_variable_fct_name("neg"))
Node.__neg__ = neg

abs_ = abs
abs = make_function(
    lambda x: abs_(x),
    pointwise_backward(lambda out, x: jnp.where(x >= 0, jnp.ones_like(x), -jnp.ones_like(x))),
    construct_single_variable_fct_name("abs"),
)
Node.__abs__ = abs

add = make_function(
    lambda a, b: a + b,
    pointwise_backward(lambda out, a, b: (1, 1)),
    construct_double_variable_fct_name("+"),
)
Node.__add__ = add
Node.__radd__ = make_scalar_dispatch_op(add)
Node.__np2bf[jnp.add] = add

sub = make_function(
    lambda a, b: a - b,
    pointwise_backward(lambda out, a, b: (1, -1)),
    construct_double_variable_fct_name("-"),
)
Node.__sub__ = sub
Node.__rsub__ = make_scalar_dispatch_op(sub)
Node.__np2bf[jnp.subtract] = sub

mul = make_function(
    lambda a, b: a * b,
    pointwise_backward(lambda out, a, b: (b, a)),
    construct_double_variable_fct_name("*"),
)
Node.__mul__ = mul
Node.__rmul__ = make_scalar_dispatch_op(mul)
Node.__np2bf[jnp.multiply] = mul

truediv = make_function(
    lambda a, b: a / b,
    pointwise_backward(lambda out, a, b: (1 / b, -a / (b * b))),
    construct_double_variable_fct_name("/"),
)
Node.__truediv__ = truediv
Node.__rtruediv__ = make_scalar_dispatch_op(truediv)
Node.__np2bf[jnp.true_divide] = truediv

floordiv = make_function(lambda a, b: a // b, name_fct=construct_double_variable_fct_name("floordiv"))
Node.__floordiv__ = floordiv
Node.__rfloordiv__ = make_scalar_dispatch_op(floordiv)
Node.__np2bf[jnp.floor_divide] = floordiv

mod = make_function(lambda a, b: a % b, name_fct=construct_double_variable_fct_name("mod"))
Node.__mod__ = mod
Node.__rmod__ = make_scalar_dispatch_op(mod)
Node.__np2bf[jnp.mod] = mod

pow = make_function(
    lambda a, b: a**b,
    pointwise_backward(lambda out, a, b: (b * a ** (b - 1), jnp.log(a) * out)),
    construct_double_variable_fct_name("pow"),
)
Node.__pow__ = pow
Node.__rpow__ = make_scalar_dispatch_op(pow)
Node.__np2bf[jnp.power] = pow

##################### LOGICAL #####################

logical_not = make_function(lambda x: ~x, name_fct=construct_single_variable_fct_name("~"))
Node.__invert__ = logical_not

logical_and = make_function(lambda a, b: a & b, name_fct=construct_double_variable_fct_name("&"))
Node.__and__ = logical_and
Node.__rand__ = make_scalar_dispatch_op(logical_and)
Node.__np2bf[jnp.logical_and] = logical_and

logical_or = make_function(lambda a, b: a | b, name_fct=construct_double_variable_fct_name("|"))
Node.__or__ = logical_or
Node.__ror__ = make_scalar_dispatch_op(logical_or)
Node.__np2bf[jnp.logical_or] = logical_or

logical_xor = make_function(lambda a, b: a ^ b, name_fct=construct_double_variable_fct_name("xor"))
Node.__xor__ = logical_xor
Node.__rxor__ = make_scalar_dispatch_op(logical_xor)
Node.__np2bf[jnp.logical_xor] = logical_xor

##################### PREDICATES #####################

eq = make_function(lambda a, b: a == b, name_fct=construct_double_variable_fct_name("=="))
Node.__eq__ = eq
Node.__np2bf[jnp.equal] = eq

ne = make_function(lambda a, b: a != b, name_fct=construct_double_variable_fct_name("!="))
Node.__ne__ = ne
Node.__np2bf[jnp.not_equal] = ne

lt = make_function(lambda a, b: a < b, name_fct=construct_double_variable_fct_name("<"))
Node.__lt__ = lt
Node.__np2bf[jnp.less] = lt

le = make_function(lambda a, b: a <= b, name_fct=construct_double_variable_fct_name("<="))
Node.__le__ = le
Node.__np2bf[jnp.less_equal] = le

gt = make_function(lambda a, b: a > b, name_fct=construct_double_variable_fct_name(">"))
Node.__gt__ = gt
Node.__np2bf[jnp.greater] = gt

ge = make_function(lambda a, b: a >= b, name_fct=construct_double_variable_fct_name(">="))
Node.__ge__ = ge
Node.__np2bf[jnp.greater_equal] = ge
