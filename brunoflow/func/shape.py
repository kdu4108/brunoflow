"""
This module defines functions that alter the shape of a tensor/tensors without changing
    their contents.
"""

import jax
from jax import numpy as jnp
from collections.abc import Iterable
from brunoflow.func.utils import (
    construct_single_variable_fct_name,
    get_relevant_out_grad_val,
    typecast_index_arg_for_jax,
)
from typing import Optional
from .function import make_function
from ..ad import Node, name

##################### RESHAPING #####################


def reshape(x, newshape):
    """
    Re-organize the values in a tensor into a tensor with a different shape

    Args:
        x: a tensor (Node or numpy.ndarray)
        newshape: a tuple of ints indicating the new shape

    Returns:
        a tensor of shape newshape containing the values from x

    Tensorflow equivalent: tf.reshape

    PyTorch equivalent: torch.reshape
    """
    return __reshape(x, newshape)


def reshape_backward(out_val, out_grad, x, newshape):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    return (
        jnp.reshape(out_grad, x.shape),
        None,
    )


__reshape = make_function(
    jnp.reshape,
    reshape_backward,
    construct_single_variable_fct_name("reshape", additional_arg_names=("shape",)),
)


def squeeze(x, axis=None):
    """
    Remove 'singleon' dimensions (i.e. dimensions of size 1) from a tensor

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: which singleton axis to remove (if None, remove all singleton axes)

    Returns:
        a tensor with the contents of x, but with singleton axes removed

    Tensorflow equivalent: tf.squeeze

    PyTorch equivalent: torch.squeeze
    """
    return _squeeze(x, axis)


def squeeze_backward(out_val, out_grad, x, axis):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    return (
        jnp.reshape(out_grad, x.shape),
        None,
    )


_squeeze = make_function(
    jnp.squeeze,
    squeeze_backward,
    construct_single_variable_fct_name("squeeze", additional_arg_names=("axis",)),
)


def expand_dims(x, axis):
    """
    Add a 'singleon' dimension (i.e. a dimension of size 1) to a tensor

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: axis at which to add a singleton dimneions

    Returns:
        a tensor with the contents of x, but with a new singleton dimension

    Tensorflow equivalent: tf.expand_dims

    PyTorch equivalent: torch.unsqueeze
    """
    return __expand_dims(x, axis)


def expand_dims_backward(out_val, out_grad, x, axis):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    return (
        jnp.reshape(out_grad, x.shape),
        None,
    )


__expand_dims = make_function(
    jnp.expand_dims,
    expand_dims_backward,
    construct_single_variable_fct_name("expand_dims", additional_arg_names=("axis",)),
)

##################### COMBINING #####################


def concat(xs, axis=0):
    """
    Concatenate multiple tensors into one tensor

    Args:
        xs: a list of tensors (Node or numpy.ndarray). Shapes must be the same, except
            along axis.
        axis: the axis along which to concatenate the tensors

    Returns:
        a single tensor formed from concatenating xs along axis

    Tensorflow equivalent: tf.concat

    PyTorch equivalent: torch.cat
    """
    return __concat(axis, *xs)


def concat_backward(out_val, out_grad, axis, *xs):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    ret_grads = [None]  # no adjoint for the 'axis' argument
    start_idx = 0
    for x in xs:
        slices = tuple(
            slice(start_idx, start_idx + dim) if i == axis else slice(0, dim) for i, dim in enumerate(x.shape)
        )
        grad = out_grad[slices]
        ret_grads.append(grad)
        start_idx += x.shape[axis]
    return ret_grads


__concat = make_function(
    lambda axis, *xs: jnp.concatenate(xs, axis),
    concat_backward,
    lambda axis, *xs: f"(concat {tuple(name(x) for x in xs)} axis={axis})",
)


def repeat(x, n, axis):
    """
    Repeat the values in a tensor multiple times to form a larger tensor

    Args:
        x: a tensor (Node or numpy.ndarray)
        n: number of times to repeat x
        axis: the axis along which to repeat x

    Returns:
        a tensor containing the values of x repeated n times along axis

    Tensorflow equivalent: tf.repeat(x, repeats=[n], axis=axis)

    PyTorch equivalent: torch.repeat_interleave(x, n, axis)
    """
    xs = [x for i in range(0, n)]
    return concat(xs, axis)


def stack(xs, axis):
    """
    Stack multiple tensors into a single tensor along a new axis

    Args:
        xs: list of tensors (Node or numpy.ndarray), all with the same shape
        axis: the new axis along which to stack xs

    Returns:
        a tensor formed from stacking xs along a new axis

    Tensorflow equivalent: tf.stack

    PyTorch equivalent: torch.stack
    """
    return concat([expand_dims(x, axis) for x in xs], axis)


##################### UN-COMBINING #####################


def get_item(x, arg):
    """
    Index into a tensor to retrieve an element or sub-tensor.

    Args:
        x: a tensor (Node or numpy.ndarray)
        arg: an indexing expression. This could be:
            * An integer index
            * A list of integer indices
            * A slice expression
            * A list of slice expressions
            * A list of both integer indices and slice expressions
            * etc.
            Any indexing expression supported by numpy is supported here.

    Returns:
        Some element / sub-tensor of x

    Tensorflow equivalent: x[arg]

    PyTorch equivalent: x[arg]
    """
    return Node.__getitem__(x, arg)


def getitem_backward(out_val, out_grad, x, arg):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    grad = jnp.zeros_like(x)
    # Need to cast list to array because otherwise:
    # `TypeError: Using a non-tuple sequence for multidimensional indexing is not allowed; use `arr[array(seq)]` instead of `arr[seq]`.
    #   See https://github.com/google/jax/issues/4564 for more information.`
    grad = grad.at[typecast_index_arg_for_jax(arg)].set(out_grad)
    return grad, None


Node.__getitem__ = make_function(
    lambda x, arg: x[
        typecast_index_arg_for_jax(arg)
    ],  # Need to cast list to array because otherwise: `TypeError: Using a non-tuple sequence for multidimensional indexing is not allowed; use `arr[array(seq)]` instead of `arr[seq]`. See https://github.com/google/jax/issues/4564 for more information.`
    getitem_backward,
    construct_single_variable_fct_name("getitem", additional_arg_names=("arg",)),
)


def _get_embedding_forward(x: Node, arg: Node, padding_idx: int):
    """
    xyzj jk
    """
    return x[typecast_index_arg_for_jax(arg)]


def _get_embedding_backward(out_val, out_grad, x, arg, padding_idx):
    out_grad = get_relevant_out_grad_val(out_grad)  # shape=(*arg.shape, emb_sz)
    grad = jnp.zeros_like(x)
    one_hot_args = jax.nn.one_hot(arg, num_classes=x.shape[0])  # shape=(*arg.shape, x.shape[0]=vocab_sz)
    transposed_one_hot_args = jnp.moveaxis(one_hot_args, -1, 0)  # shape=(vocab_sz, *arg.shape)

    grad = jnp.einsum(
        "i...,...l->il", transposed_one_hot_args, out_grad
    )  # shape=(vocab_sz, emb_sz) (i=vocab_sz, l=emb_sz, the ... = arg.shape which is shared)

    if padding_idx is not None:
        grad = grad.at[padding_idx].set(0.0)

    # return 0s for arg because it may sometimes be a Node (e.g. when loading position_ids in BfBertEmbedding) and we need to pass it some sort of grad back.
    return grad, jnp.zeros_like(arg), None


__get_embedding = make_function(
    _get_embedding_forward,
    _get_embedding_backward,
    lambda x, arg, padding_idx: "get_embedding",
)


def get_embedding(x: Node, arg: Node, padding_idx: Optional[int] = None):
    """
    Gets the `arg` indices from x while always ensuring that the gradient at the `padding_idx`-th row is 0.
    This is basically getitem but useful for embeddings because of the enforced 0-ing of grad for padding_idx.
    """

    return __get_embedding(x, arg, padding_idx)


def split(x, indices_or_sections, axis):
    """
    Split a tensor into multiple sub-tensors

    Args:
        x: a tensor (Node or numpy.ndarray)
        indices_or_sections: either:
            * An integer indicating the number of splits to make
            * A list of integers indicating indices where splits should be made
        axis: axis along which to split x

    Returns:
        a list of tensors formed by splitting x according to the args.

    Tensorflow equivalent: tf.split
        * Note that the second argument to tf.split expects *sizes* of splits, not indices where they occur

    PyTorch equivalent: torch.split
        * Note that the second argument to torch.split expects *sizes* of splits, not indices where they occur
    """
    if isinstance(indices_or_sections, int):
        N = indices_or_sections
        size = x.shape[axis]
        if size % N != 0:
            raise ValueError(
                f"Size of array along axis must be divisible by number of splits; {size} is not divisible by {N}"
            )
        increment = size // N
        indices_or_sections = [i * increment for i in range(1, N)]
    if isinstance(indices_or_sections, Iterable):
        indices = indices_or_sections
        indices.sort()
        if not all([isinstance(i, int) for i in indices]):
            raise TypeError("indices argument to split must contain only ints")
        indices.insert(0, 0)
        indices.append(x.shape[axis])
        splits = []
        for i in range(0, len(indices) - 1):
            slices = []
            for j in range(x.ndim):
                start = indices[i] if j == axis else 0
                stop = indices[i + 1] if j == axis else x.shape[j]
                slices.append(slice(start, stop))
            splits.append(x[tuple(slices)])
        return splits
    else:
        raise TypeError(f"Argument to split must be either int or Iterable; got {type(indices_or_sections)} instead")


def unstack(x, axis):
    """
    Unpacks a rank N tensor into multiple rank (N-1) tensors along a given axis.
    This is the inverse of the 'stack' function

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: axis along which to unpack x

    Returns:
        a list of the sub-tensors of x along axis

    Tensorflow equivalent: tf.unstack

    PyTorch equivalent: torch.unbind
    """
    N = x.shape[axis]
    splits = split(x, N, axis)
    return [squeeze(y) for y in splits]
