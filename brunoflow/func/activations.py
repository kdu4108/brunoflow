"""
This module defines activation functions commonly used in neural networks

Note that tanh is not defined here, but rather in math.py

For some of these functions, we've implemented them as compositions of
    existing Bluenoflow functions.
This is correct and easy to do, but it is less efficient than implementing
    a new 'atomic' autodiff function via make_function (since each call
    to such an activation function will add multiple nodes to the
    computation graph).
"""

import jax
from jax import numpy as jnp
from .function import make_function, pointwise_backward
from . import math
from .reductions import reduce_logsumexp
from .shape import expand_dims
from brunoflow.func.utils import construct_single_variable_fct_name


def sigmoid(x):
    """
    The sigmoid activation function: 1 / (1 + e^(-x))

    Args:
        x: a tensor (a Node or numpy.ndarray)

    Returns:
        a tensor of the same shape as x with values in the range [0, 1]

    Tensorflow equivalent: tf.math.sigmoid

    PyTorch equivalent: torch.nn.functional.sigmoid
    """
    return __sigmoid(x)


__sigmoid = make_function(
    lambda x: 1 / (1 + jnp.exp(-x)),
    pointwise_backward(lambda out, x: out * (1 - out)),
    construct_single_variable_fct_name("sigmoid"),
)


def softplus(x):
    """
    The softpus activation function: log(1 + e^x)

    Args:
        x: a tensor (a Node or numpy.ndarray)

    Returns:
        a tensor of the same shape as x with values in the range [0, inf]

    Tensorflow equivalent: tf.math.softplus

    PyTorch equivalent: torch.nn.functional.softplus
    """
    return __softplus(x)


__softplus = make_function(
    lambda x: jnp.log(1 + jnp.exp(x)),
    pointwise_backward(lambda out, x: jnp.exp(x) / (1 + jnp.exp(x))),
    construct_single_variable_fct_name("softplus"),
)


def relu(x):
    """
    The ReLU activation function: max(x, 0)

    Args:
        x: a tensor (a Node or numpy.ndarray)

    Returns:
        a tensor of the same shape as x with values in the range [0, inf]

    Tensorflow equivalent: tf.nn.softplus

    PyTorch equivalent: torch.nn.functional.relu
    """
    return math.maximum(x, 0)


def leakyrelu(x, neg_slope=0.01):
    """
    The Leaky ReLU activation function: max(x, 0) + neg_slope*min(x, 0)

    Args:
        x: a tensor (a Node or numpy.ndarray)

    Returns:
        a tensor of the same shape as x with values in the range [-inf, inf]

    Tensorflow equivalent: tf.nn.leaky_relu

    PyTorch equivalent: torch.nn.functional.leaky_relu
    """
    return _leakyrelu(x, neg_slope)
    # return relu(x) + math.minimum(x, 0) * neg_slope


def leakyrelu_backward(out_val, out_grad, x, neg_slope=0.01):
    return jnp.where(out_val > 0, 1, neg_slope), None


_leakyrelu = make_function(
    lambda x, neg_slope: jnp.maximum(x, 0) + jnp.minimum(x, 0) * neg_slope,
    pointwise_backward(leakyrelu_backward),
    construct_single_variable_fct_name("leakyrelu", additional_arg_names=("neg_slope",)),
)


@jax.jit
def _gelu_exact(x):
    return jax.nn.gelu(x, approximate=False)


@jax.jit
def gelu_exact_backward(out, x):
    jac = jax.jacfwd(_gelu_exact)(x)
    return jnp.sum(jac, axis=tuple(range(len(jac.shape) - len(x.shape))))


gelu = make_function(_gelu_exact, pointwise_backward(gelu_exact_backward), name_fct=lambda x: "gelu")


@jax.jit
def _gelu_approx(x):
    return jax.nn.gelu(x, approximate=True)


@jax.jit
def gelu_approx_backward(out, x):
    jac = jax.jacfwd(_gelu_approx)(x)
    return jnp.sum(jac, axis=tuple(range(len(jac.shape) - len(x.shape))))


gelu_approx = make_function(_gelu_approx, pointwise_backward(gelu_approx_backward), name_fct=lambda x: "gelu_approx")


def softmax(x, axis):
    """
    The softmax function: e^x / sum(e^x)

    Uses an efficient and nnumerically-stable implementation.

    Args:
        x: a tensor of logits (a Node or numpy.ndarray)
        axis: which axis of x should be treated as the logits

    Returns:
        a tensor of the same shape as x whose values along axis are probabilities

    Tensorflow equivalent: tf.nn.softmax

    PyTorch equivalent: torch.nn.functional.softmax
    """
    return math.exp(log_softmax(x, axis=axis))


def log_softmax(x, axis):
    """
    The log softmax function: log(e^x / sum(e^x))

    Uses an efficient and nnumerically-stable implementation.

    Args:
        x: a tensor of logits (a Node or numpy.ndarray)
        axis: which axis of x should be treated as the logits

    Returns:
        a tensor of the same shape as x whose values along axis are log-probabilities

    Tensorflow equivalent: tf.nn.log_softmax

    PyTorch equivalent: torch.nn.functional.log_softmax
    """
    return x - expand_dims(reduce_logsumexp(x, axis=axis), axis)


ACT2FN = {
    "gelu": gelu,
    "gelu_new": gelu_approx,
}
