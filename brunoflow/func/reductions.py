"""
This module defines functions that perform tensor reductions
"""
import jax
from jax import numpy as jnp
from brunoflow.ad import name
from brunoflow.func.utils import construct_single_variable_fct_name, custom_put_along_axis, get_relevant_out_grad_val
from .function import make_function
from . import math
from .shape import expand_dims


def reduce_min(x, axis=None):
    """
    Compute the minimum value of a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the minimum value of x
        If axis is not None: a tensor with one fewer dimension than x containing the
            minimum values of x along axis

    Tensorflow equivalent: tf.math.reduce_min

    PyTorch equivalent: torch.min
    """
    return _reduce_min(x, axis)


def reduce_min_backward(out_val, out_grad, x, axis=None):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    grad = jnp.zeros_like(x)
    if axis is None:
        min_index = jnp.argmin(x)
        min_index = jnp.unravel_index(min_index, x.shape)
        grad = grad.at[min_index].set(out_grad)
    else:
        min_indices = jnp.argmin(x, axis)
        min_indices = jnp.expand_dims(min_indices, axis)
        out_grad = jnp.expand_dims(out_grad, axis)
        grad = custom_put_along_axis(grad, min_indices, out_grad, axis)
    return grad, None


_reduce_min = make_function(
    lambda x, axis: x.min(axis=axis),
    reduce_min_backward,
    construct_single_variable_fct_name("min", additional_arg_names=("axis",)),
)


def reduce_max(x, axis=None):
    """
    Compute the maximium value of a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the maximum value of x
        If axis is not None: a tensor with one fewer dimension than x containing the
            maximum values of x along axis

    Tensorflow equivalent: tf.math.reduce_max

    PyTorch equivalent: torch.max
    """
    return _reduce_max(x, axis)


def reduce_max_backward(out_val, out_grad, x, axis=None):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    grad = jnp.zeros_like(x)
    if axis is None:
        min_index = jnp.argmax(x)
        min_index = jnp.unravel_index(min_index, x.shape)
        grad = grad.at[min_index].set(out_grad)
    else:
        min_indices = jnp.argmax(x, axis)
        min_indices = jnp.expand_dims(min_indices, axis)
        out_grad = jnp.expand_dims(out_grad, axis)
        grad = custom_put_along_axis(grad, min_indices, out_grad, axis)
    return grad, None


_reduce_max = make_function(
    lambda x, axis: x.max(axis=axis),
    reduce_max_backward,
    construct_single_variable_fct_name("max", additional_arg_names=("axis",)),
)


def reduce_sum(x, axis=None, keepdims=None):
    """
    Compute the sum of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the sum of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            sum of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_sum

    PyTorch equivalent: torch.sum
    """
    return _reduce_sum(x, axis, keepdims)


def reduce_sum_backward(out_val, out_grad, x, axis=None, keepdims=None):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.

    # TODO: verify whether this...actually works?
    out_grad = get_relevant_out_grad_val(out_grad)
    if not keepdims:
        out_grad = jnp.expand_dims(out_grad, axis if axis is not None else list(range(len(x))))
    jac = jax.jacfwd(jnp.sum)(x, axis=axis, keepdims=True)
    return (
        out_grad * jnp.sum(jac, axis=tuple(range(len(jac.shape) - len(x.shape)))),
        None,
        None,
    )
    # if axis is None:
    #     return jnp.full(x.shape, out_grad), None, None
    # else:
    #     if isinstance(axis, int):
    #         axis = (axis,)
    #     res = out_grad if not keepdims else out_grad[axis]
    #     for ax in axis:
    #         res = jnp.stack([res for _ in range(x.shape[ax])], axis=ax)
    #     return res, None, None

    #     # stacks = [out_grad for i in range(x.shape[axis])]
    #     # return jnp.stack(stacks, axis), None


_reduce_sum = make_function(
    lambda x, axis, keepdims: jnp.sum(x, axis=axis, keepdims=keepdims),
    reduce_sum_backward,
    construct_single_variable_fct_name("sum", additional_arg_names=("axis", "keepdims")),
)


def reduce_mean(x, axis=None, keepdims=None):
    """
    Compute the mean of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the mean of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            mean of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_mean

    PyTorch equivalent: torch.mean
    """
    if axis is None:
        n = x.size
    elif isinstance(axis, int):
        n = x.shape[axis]
    elif isinstance(axis, tuple):
        n = jnp.product(jnp.array([x.shape[ax] for ax in axis]))

    return reduce_sum(x, axis, keepdims=keepdims) / n


def reduce_var(x, axis=None):
    """
    Compute the variance of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the variance of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            variance of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_variance

    PyTorch equivalent: torch.var
    """
    mean = reduce_mean(x, axis)
    if axis is None:
        diff = x - mean
    else:
        diff = x - expand_dims(mean, axis)
    return reduce_mean(diff * diff, axis)


def reduce_std(x, axis=None):
    """
    Compute the standard deviation of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the standard deviation of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            standard deviation of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_std

    PyTorch equivalent: torch.std
    """
    return math.sqrt(reduce_var(x, axis))


def reduce_logsumexp(x, axis=None):
    """
    Compute the log of the sum of the exponent of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the log-sum-exp of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            log-sum-exp of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_logsumexp

    PyTorch equivalent: torch.logsumexp
    """
    max_val = reduce_max(x, axis)
    if axis is None:
        exps = math.exp(x - max_val)
    else:
        exps = math.exp(x - expand_dims(max_val, axis))
    return math.log(reduce_sum(exps, axis)) + max_val
    # return math.log(reduce_sum(math.exp(x)))
