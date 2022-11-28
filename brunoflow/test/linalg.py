import brunoflow as bf
import jax
from jax import numpy as jnp
import torch
import scipy.stats as ss
import unittest as ut

from . import utils


class LinalgTestCase(ut.TestCase):
    # To run: `XLA_PYTHON_CLIENT_PREALLOCATE=false python -m unittest brunoflow.test.linalg.LinalgTestCase`, see # https://github.com/google/jax/issues/7118#issuecomment-950183972

    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)

    def test_matmul_1x1_abs_val_grad_and_entropy(self):
        x_bf = bf.Node(jnp.array([[2]]))
        y_bf = bf.Node(jnp.array([[-9]]))
        out = x_bf @ y_bf
        out.backprop()
        self.assertTrue(jnp.array_equal(x_bf.grad, [[-9]]))
        self.assertTrue(jnp.array_equal(y_bf.grad, [[2]]))
        self.assertTrue(jnp.array_equal(x_bf.abs_val_grad, [[9]]))
        self.assertTrue(jnp.array_equal(y_bf.abs_val_grad, [[2]]))
        self.assertTrue(jnp.array_equal(x_bf.compute_entropy().val, [[0]]))
        self.assertTrue(jnp.array_equal(y_bf.compute_entropy().val, [[0]]))

    def test_matmul_1x1_abs_val_grad_and_entropy_when_input_has_val_0_isnan(self):
        # This isn't actually something we want, but unclear what the right behavior is.
        x_bf = bf.Node(jnp.array([[0]]))
        y_bf = bf.Node(jnp.array([[-9]]))
        out = x_bf @ y_bf
        out.backprop()
        self.assertTrue(jnp.array_equal(x_bf.grad, [[-9]]))
        self.assertTrue(jnp.array_equal(y_bf.grad, [[0]]))
        self.assertTrue(jnp.array_equal(x_bf.abs_val_grad, [[9]]))
        self.assertTrue(jnp.array_equal(y_bf.abs_val_grad, [[0]]))
        self.assertTrue(jnp.array_equal(x_bf.compute_entropy().val, [[0]]))
        self.assertTrue(jnp.isnan(y_bf.compute_entropy().val))

    def test_matmul_1x2_abs_val_grad_and_entropy(self):
        x_bf = bf.Node(jnp.array([[1, 2]]))
        y_bf = bf.Node(jnp.array([[3], [-9]]))
        out = x_bf @ y_bf
        out.backprop()
        self.assertTrue(jnp.array_equal(x_bf.grad, jnp.array([[3, -9]])))
        self.assertTrue(jnp.array_equal(y_bf.grad, jnp.array([[1], [2]])))
        self.assertTrue(jnp.array_equal(x_bf.abs_val_grad, jnp.array([[3, 9]])))
        self.assertTrue(jnp.array_equal(y_bf.abs_val_grad, jnp.array([[1], [2]])))
        self.assertTrue(jnp.array_equal(x_bf.compute_entropy().val, jnp.array([[0.0, 0.0]])))
        self.assertTrue(jnp.array_equal(y_bf.compute_entropy().val, jnp.array([[0.0], [0.0]])))

    def test_matmul_1_times_1x2_abs_val_grad_and_entropy(self):
        x_bf = bf.Node(jnp.array([[2]]))
        y_bf = bf.Node(jnp.array([[3, -9]]))
        out = x_bf @ y_bf
        out.backprop()

        # "out_grad" here is simply the identity [[1], [1]], so the "weighted sum" of how the different components of y contributes to the derivative w.r.t. to x is just the sum?
        self.assertTrue(jnp.array_equal(x_bf.grad, jnp.array([[-6]])))
        self.assertTrue(jnp.array_equal(y_bf.grad, jnp.array([[2, 2]])))
        self.assertTrue(jnp.array_equal(x_bf.abs_val_grad, jnp.array([[12]])))
        self.assertTrue(jnp.array_equal(y_bf.abs_val_grad, jnp.array([[2, 2]])))
        self.assertTrue(jnp.allclose(x_bf.compute_entropy().val, jnp.array([[ss.entropy([0.25, 0.75])]])))
        self.assertTrue(jnp.array_equal(y_bf.compute_entropy().val, jnp.array([[0, 0]])))

    def test_matmul_2x1_times_1x2_abs_val_grad_and_entropy(self):
        x_bf = bf.Node(jnp.array([[2], [8]]))
        y_bf = bf.Node(jnp.array([[3, -9]]))
        out = x_bf @ y_bf
        out.backprop()

        # "out_grad" here is simply the identity [[1, 1], [1, 1]], so the "weighted sum" of how the different components of y contributes to the derivative w.r.t. to x is just the sum?
        self.assertTrue(jnp.array_equal(x_bf.grad, jnp.array([[-6], [-6]])))
        self.assertTrue(jnp.array_equal(y_bf.grad, jnp.array([[10, 10]])))
        self.assertTrue(jnp.array_equal(x_bf.abs_val_grad, jnp.array([[12], [12]])))
        self.assertTrue(jnp.array_equal(y_bf.abs_val_grad, jnp.array([[10, 10]])))
        self.assertTrue(
            jnp.allclose(x_bf.compute_entropy().val, jnp.array([[ss.entropy([0.25, 0.75]), ss.entropy([0.25, 0.75])]]))
        )
        self.assertTrue(
            jnp.allclose(y_bf.compute_entropy().val, jnp.array([[ss.entropy([0.2, 0.8]), ss.entropy([0.2, 0.8])]]))
        )


######################################################

utils.add_tests(
    LinalgTestCase,
    basename="test_transpose",
    fn=lambda x: bf.transpose(x, [1, 2, 0]),
    torch_fn=lambda x: x.permute(1, 2, 0),
    inputs=utils.random_inputs([[(5, 4, 3)]]),
)

utils.add_tests(
    LinalgTestCase,
    basename="test_matrix_transpose",
    fn=lambda x: bf.matrix_transpose(x),
    torch_fn=lambda x: torch.transpose(x, -2, -1),
    inputs=utils.random_inputs([[(4, 4)], [(5, 4)], [(3, 5, 4)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_diag_k=0",
    fn=lambda x: bf.diag(x),
    torch_fn=lambda x: torch.diagonal(x, dim1=-2, dim2=-1),
    inputs=utils.random_inputs([[(4, 4)], [(5, 4)], [(4, 5)], [(3, 5, 4)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_diag_k=1",
    fn=lambda x: bf.diag(x, k=1),
    torch_fn=lambda x: torch.diagonal(x, offset=1, dim1=-2, dim2=-1),
    inputs=utils.random_inputs([[(4, 4)], [(5, 4)], [(4, 5)], [(3, 5, 4)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_diag_k=-1",
    fn=lambda x: bf.diag(x, k=-1),
    torch_fn=lambda x: torch.diagonal(x, offset=-1, dim1=-2, dim2=-1),
    inputs=utils.random_inputs([[(4, 4)], [(5, 4)], [(4, 5)], [(3, 5, 4)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_trace",
    fn=lambda x: bf.trace(x),
    torch_fn=lambda x: torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1),
    inputs=utils.random_inputs([[(4, 4)], [(5, 4)], [(4, 5)], [(3, 5, 4)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_det",
    fn=lambda x: bf.det(x),
    torch_fn=lambda x: torch.det(x),
    inputs=utils.random_inputs([[(4, 4)], [(3, 5, 5)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_inv",
    fn=lambda x: bf.inv(x),
    torch_fn=lambda x: torch.inverse(x),
    inputs=utils.random_inputs([[(4, 4)], [(3, 5, 5)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_norm_all",
    fn=lambda x: bf.norm(x),
    torch_fn=lambda x: torch.norm(x),
    inputs=utils.random_inputs(
        [
            [(5)],
            [(4, 4)],
        ]
    ),
)

utils.add_tests(
    LinalgTestCase,
    basename="test_norm_axis_vector",
    fn=lambda x: bf.norm(x, axis=-1),
    torch_fn=lambda x: torch.norm(x, dim=-1),
    inputs=utils.random_inputs([[(3, 5)]]),  # Batch of vectors
)

utils.add_tests(
    LinalgTestCase,
    basename="test_norm_axis_matrix",
    fn=lambda x: bf.norm(x, axis=(-2, -1)),
    torch_fn=lambda x: torch.norm(x, dim=(-2, -1)),
    inputs=utils.random_inputs([[(3, 5, 5)]]),  # Batch of matrices
)

utils.add_tests(
    LinalgTestCase,
    basename="test_matmul",
    fn=lambda x, y: bf.matmul(x, y),
    torch_fn=lambda x, y: torch.matmul(x, y),
    inputs=utils.random_inputs(
        [
            [(4, 4), (4, 4)],
            [(3, 4), (4, 5)],
            [(2, 4, 4), (2, 4, 4)],
            [(2, 3, 4), (2, 4, 5)],
            [(2, 4, 4), (4, 4)],
            [(3, 4), (2, 4, 5)],
        ]
    ),
)

utils.add_tests(
    LinalgTestCase,
    basename="test_matmul_op",
    fn=lambda x, y: x @ y,
    torch_fn=lambda x, y: x @ y,
    inputs=utils.random_inputs(
        [
            [(4, 4), (4, 4)],
            [(3, 4), (4, 5)],
            [(2, 4, 4), (2, 4, 4)],
            [(2, 3, 4), (2, 4, 5)],
            [(2, 4, 4), (4, 4)],
            [(3, 4), (2, 4, 5)],
        ]
    ),
)
