"""
This module implements autodiff versions of common linear algebra functions.
"""

import numpy as np
from .function import make_function
from .reductions import *
from . import math
from ..ad import Node
from brunoflow.func.utils import (
    construct_single_variable_fct_name,
    construct_double_variable_fct_name,
    get_relevant_out_grad_val,
)


def transpose(x, axes):
    """
    Permute the axes of a tensor

    Tensorflow equivalent: tf.transpose(x, axes)

    PyTorch equivalent: torch.Tensor.permute(x, axes)
    """
    return __transpose(x, axes)


def transpose_backward(out_val, out_grad, x, axes):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    inverse_perm = np.arange(len(axes))[np.argsort(axes)]
    return np.transpose(out_grad, inverse_perm), None


__transpose = make_function(
    lambda x, axes: np.transpose(x, axes),
    transpose_backward,
    construct_single_variable_fct_name("transpose", ("axes",)),
)


def matrix_transpose(x):
    """
    Permute the last two axes of a tensor

    Tensorflow equivalent: tf.linalg.matrix_transpose(x)

    PyTorch equivalent: torch.transpose(x, -1, -2)
    """
    axes = list(range(x.ndim))
    axes[-2] = x.ndim - 1
    axes[-1] = x.ndim - 2
    return transpose(x, axes)


def __np_matrix_transpose(x):
    """
    A numpy version of matrix_transpose, since numpy lacks such a transpose function
        that works on batches of matrices.
    This is used as a helper function later on in this module
    """
    axes = list(range(x.ndim))
    axes[-2] = x.ndim - 1
    axes[-1] = x.ndim - 2
    return np.transpose(x, axes)


def diag(x, k=0):
    """
    Extract the k-th diagonal of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.diag_part(x, k)

    PyTorch equivalent: torch.diagonal(x, k, dim1=-2, dim2=-1),
    """
    return _diag(x, k)


def diag_backward(out_val, out_grad, x, k):
    """
    This function is used to compute d(output)/d(input_to_diag), aka the gradient of the output node (e.g. loss) w.r.t. the input of the diag function.
    It is NOT in the form of d(diag)/d(input_to_diag) * d(output)/d(diag) (aka local_gradient * out_grad) because of weird shape things?
    We can compute it directly though, as shown here.

    This function is also used to compute d_abs(output)/d_abs(input_to_diag) by the upstream function passing in
        diag_node.abs_val_grad for out_grad (instead of diag_node.grad). This is safe here because, by inspection of the function, we have the invariant that
        (diag_node.abs_val_grad >= 0) ==> (diag_backward(diag_node.abs_val_grad) >= 0).
    """
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    if x.ndim == 2:
        return diagflat_nonsquare(out_grad, k, x.shape)
    else:
        out_grad = np.reshape(out_grad, (-1, out_grad.shape[-1]))
        b = out_grad.shape[0]
        n = x.shape[-2]
        m = x.shape[-1]
        grad = np.zeros((b, n, m))
        for i in range(b):
            grad[i] = diagflat_nonsquare(out_grad[i], k, grad[i].shape)
        return np.reshape(grad, x.shape), None


def diagflat_nonsquare(diag_vec, k, out_shape):
    diag_mat = np.diagflat(diag_vec, k)
    ds = diag_mat.shape
    os = out_shape
    # Add extra padding rows/cols, if needed
    diag_mat = np.pad(diag_mat, [(0, max(0, os[0] - ds[0])), (0, max(0, os[1] - ds[1]))], "constant")
    # Delete any extraneous rows/cols, if needed
    diag_mat = diag_mat[0 : os[0], 0 : os[1]]
    return diag_mat


_diag = make_function(
    lambda x, k: np.diagonal(x, offset=k, axis1=-2, axis2=-1),
    diag_backward,
    construct_single_variable_fct_name("diag", additional_arg_names=("offset",)),
)


def trace(x):
    """
    Compute the trace of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.trace(x)

    PyTorch equivalenet: torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
    """
    return reduce_sum(diag(x, k=0), axis=-1)


def det(x):
    """
    Compute the determinant of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.det(x)

    PyTorch equivalent: torch.det(x)
    """
    return __det(x)


# This function is used to compute d(output)/d(input_to_det), aka the gradient of the output node (e.g. loss) w.r.t. the input of the det function.
# It is NOT in the form of d(det)/d(input_to_det) * d(output)/d(det) (aka local_gradient * out_grad) because of weird shape things?
# We can compute it directly though, as shown here.

# This function is also used to compute d_abs(output)/d_abs(input_to_det) by the upstream function passing in
#     det_node.abs_val_grad for out_grad (instead of det_node.grad). This is safe here because, by inspection of the function, we have the invariant that
#     (det_node.abs_val_grad >= 0) ==> (det_backward(det_node.abs_val_grad) >= 0).
def det_backward(out_val, out_grad, x):
    # In this function, the value of out_grad represents the upstream accumulated value for a semiring.
    #  So, out_grad may be an int representing the accumulated semiring value (e.g. gradient)
    #  or, if the semiring is more complicated, a dict containing the components of the accumulated
    #  semiring value (e.g. the abs_val_grad and entropy).
    # Extract the appropriate value for the semiring.
    out_grad = get_relevant_out_grad_val(out_grad)
    return np.expand_dims(np.expand_dims(out_val * out_grad, -1), -1) * __np_matrix_transpose(np.linalg.inv(x))


__det = make_function(
    lambda x: np.linalg.det(x),
    det_backward,
    construct_single_variable_fct_name("det"),
)


def inv(x):
    """
    Compute the inverse of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.inv(x)

    PyTorch equivalent: torch.inverse(x)
    """
    return __inv(x)


def inv_backward(out_val, out_grad, x):
    """This might not be safe with entropy/abs val computations?"""
    # TODO: fix this one!
    if isinstance(out_grad, dict):
        raise ValueError(
            "ERROR: calling the backward pass of inv to compute entropy and abs_val_grad of a graph containing this `bf.linalg.inv` function may "
            "not be safe because the output of the backward pass may be negative, even if out_grad=inv_node.abs_val_grad is positive. "
            "If this node is required as part of a computation graph, then refactor code so that this computes "
            "d_abs(output)/d_abs(input_to_inv) s.t. it MUST be non-negative."
        )
    out_grad = get_relevant_out_grad_val(out_grad)
    out_val_T = __np_matrix_transpose(out_val)
    return -np.matmul(out_val_T, np.matmul(out_grad, out_val_T))


__inv = make_function(lambda x: np.linalg.inv(x), inv_backward, construct_single_variable_fct_name("inv"))


def norm(x, axis=None):
    """
    Compute the norm of a (batch of) vector or matrix

    If axis is an int, this function interprets this axis of x as corresponding to a vector,
        and it computes the L2 norm of this vector.

    If axis is a tuple (int, int), this function interprets these two axes of x as
        corresponding to a matrix, and it computes the Frobenius norm of this matrix

    Tensorflow equivalent:
        tf.norm(x, 'euclidean', axis=-1)        [for vector norm]
        tf.norm(x, 'euclidean', axis=[-2, -1])  [for matrix norm]

    PyTorch equivalent:
        torch.norm(x, None, dim=-1)         [for vector norm]
        torch.norm(x, 'fro', dim=(-1, -2))  [for matrix norm]

    """
    x_2 = x * x
    if isinstance(axis, tuple):
        assert len(axis) == 2
        a0, a1 = axis
        # Convert to absolute indices, if they are negative
        if a0 < 0:
            a0 = x.ndim + a0
        if a1 < 0:
            a1 = x.ndim + a1
        x_red = reduce_sum(x_2, axis=a0)
        if a1 > a0:
            a1 -= 1
        x_red = reduce_sum(x_red, axis=a1)
    else:
        x_red = reduce_sum(x_2, axis=axis)
    return math.sqrt(x_red)


# Matrix multiplication and its infix operator (@)
def matmul_backward(out_val, out_grad, A, B):
    A_factor = np.copy(A)
    B_factor = np.copy(B)
    if isinstance(out_grad, dict) and "out_entropy" in out_grad:
        out_entropy = out_grad["out_entropy"]
        out_abs_val_grad = out_grad["out_abs_val_grad"]
        A_factor = np.abs(A_factor)
        B_factor = np.abs(B_factor)

        return (
            np.matmul(out_entropy, __np_matrix_transpose(B_factor))
            + np.matmul(out_abs_val_grad, __np_matrix_transpose(np.multiply(-B_factor, np.log(B_factor)))),
            np.matmul(__np_matrix_transpose(A_factor), out_entropy)
            + np.matmul(__np_matrix_transpose(np.multiply(-A_factor, np.log(A_factor))), out_abs_val_grad),
        )

    elif isinstance(out_grad, dict):
        out_grad = out_grad["out_abs_val_grad"]
        A_factor = np.abs(A_factor)
        B_factor = np.abs(B_factor)

    return (
        np.matmul(out_grad, __np_matrix_transpose(B_factor)),
        np.matmul(__np_matrix_transpose(A_factor), out_grad),
    )


matmul = make_function(
    lambda A, B: np.matmul(A, B),
    matmul_backward,
    construct_double_variable_fct_name("matmul"),
)
Node.__matmul__ = matmul
Node.__rmatmul__ = lambda A, B: Node.__mul__(B, A)
