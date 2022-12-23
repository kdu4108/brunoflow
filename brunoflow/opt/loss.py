"""
This module implements common loss functions
"""
import jax
from jax import grad, numpy as jnp
import scipy.special as ssp

from ..func import math, make_function, pointwise_backward
from ..func.shape import reshape
from ..func.reductions import reduce_sum, reduce_mean
from ..func.activations import log_softmax
from .. import ad


def mse_loss(output, target, reduction="mean"):
    """
    Compute the squared difference between an output and a target value

    Args:
        output: a tensor (Node or numpy.ndarray)
        target: a tensor (Node or numpy.ndarray)
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce((output - target)^2), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.compat.v1.losses.mean_squared_error

    PyTorch equivalent: torch.nn.functional.mse_loss
    """
    diff = output - target
    err = diff * diff
    if reduction == "none":
        return err
    elif reduction == "mean":
        return reduce_mean(err)
    elif reduction == "sum":
        return reduce_sum(err)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def l1_loss(output, target, reduction="mean"):
    """
    Compute the absolute difference between an output and a target value

    Args:
        output: a tensor (Node or numpy.ndarray)
        target: a tensor (Node or numpy.ndarray)
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(|output - target|), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.compat.v1.losses.absolute_difference

    PyTorch equivalent: torch.nn.functional.l1_loss
    """
    err = abs(output - target)
    if reduction == "none":
        return err
    elif reduction == "mean":
        return reduce_mean(err)
    elif reduction == "sum":
        return reduce_sum(err)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def bce_loss(output, target, reduction="mean"):
    """
    Compute the binary cross entropy between an output and a target value

    Args:
        output: a tensor (Node or numpy.ndarray)
        target: a tensor (Node or numpy.ndarray)
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(bce(output, target)), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.backend.binary_crossentropy

    PyTorch equivalent: torch.nn.functional.binary_cross_entropy
    """
    err = -(math.log(output) * target + math.log(1.0 - output) * (1.0 - target))
    if reduction == "none":
        return err
    elif reduction == "mean":
        return reduce_mean(err)
    elif reduction == "sum":
        return reduce_sum(err)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def nll_loss(output, target, reduction="mean"):
    """
    Compute the negative log likelihood of a target class under an output log probability
        distribution.

    Args:
        output: a tensor (Node or numpy.ndarray) containing log probabilities
        target: a numpy.ndarray with dtype=int containing class labels
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(-output[target]), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.losses.categorical_crossentropy(tf.one_hot(target), tf.log(output))

    PyTorch equivalent: torch.nn.functional.nll_loss
    """
    target = ad.value(target)
    shape = target.shape
    output = reshape(output, [-1, output.shape[-1]])
    target = jnp.reshape(target, [-1])
    err = -output[jnp.arange(target.size), target]
    if reduction == "none":
        return reshape(err, shape)
    elif reduction == "mean":
        return reduce_mean(err)
    elif reduction == "sum":
        return reduce_sum(err)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def cross_entropy_loss(output, target, reduction="mean"):
    """
    Compute the cross entropy of logits w.r.t. a target class.
    This is just log softmax followed by the NLL loss.

    Args:
        output: a tensor (Node or numpy.ndarray) containing logits
        target: a numpy.ndarray with dtype=int containing class labels
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(nll_loss(log_softmax(output), target)), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.losses.categorical_crossentropy(tf.one_hot(target), output, from_logits=True)

    PyTorch equivalent: torch.nn.functional.cross_entropy
    """
    return nll_loss(log_softmax(output, -1), target, reduction)


def cross_entropy_loss2(output, target, reduction="mean"):
    """
    Compute the cross entropy of logits w.r.t. a target class.
    This is just log softmax followed by the NLL loss.

    Args:
        output: a tensor (Node or numpy.ndarray) containing logits
        target: a numpy.ndarray with dtype=int containing class labels
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(nll_loss(log_softmax(output), target)), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.losses.categorical_crossentropy(tf.one_hot(target), output, from_logits=True)

    PyTorch equivalent: torch.nn.functional.cross_entropy
    """

    def nll_loss(output, target, reduction):
        target = ad.value(target)
        shape = target.shape
        output = jnp.reshape(output, [-1, output.shape[-1]])
        target = jnp.reshape(target, [-1])
        err = -output[jnp.arange(target.size), target]
        if reduction == "none":
            return jnp.reshape(err, shape)
        elif reduction == "mean":
            return jnp.mean(err)
        elif reduction == "sum":
            return jnp.sum(err)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")

    return nll_loss(jax.nn.log_softmax(output, axis=-1), target, reduction)


# def _cross_entropy_loss_backward(out, logits, target, reduction="mean"):
#     """
#     out - the output of the forward pass
#     logits - the logit input value to the forward pass
#     target - the target input value to the forward pass
#     """
#     m = target.shape[0]
#     probs = ssp.softmax(ad.value(logits), axis=-1)
#     err = probs
#     err[range(m), target] -= 1
#     return err, None, None

_cross_entropy_loss_backward = grad(cross_entropy_loss2)

cross_entropy_loss_raw = make_function(
    forward=cross_entropy_loss2,
    backward=pointwise_backward(lambda out_val, output, target: (_cross_entropy_loss_backward(output, target), None)),
    name_fct=lambda a, b: f"(cross_entropy_loss {ad.name(a)})",
)


def regularize(model, l1_weight=0, l2_weight=1):
    regularization_term = 0
    l1_penalty = 0
    l2_penalty = 0

    if l1_weight != 0:
        for p in model.parameters():
            l1_penalty += reduce_sum(abs(p))
        l1_penalty.set_name("L1_penalty")
        regularization_term += l1_penalty * l1_weight

    if l2_weight != 0:
        for p in model.parameters():
            l2_penalty += reduce_sum(p**2)
        l2_penalty.set_name("L2_penalty")
        regularization_term += l2_penalty * l2_weight

    return regularization_term
