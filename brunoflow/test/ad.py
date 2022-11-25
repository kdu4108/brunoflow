from typing_extensions import assert_type
import brunoflow as bf
import jax
from jax import numpy as jnp
import scipy.stats as ss
import unittest as ut

from brunoflow.ad import name
from brunoflow.ad.node import Node
from brunoflow.func.utils import print_vals_for_all_children
from brunoflow.opt import regularize
from brunoflow.test.utils import get_all_paths_from_src_to_target_node, get_all_paths_to_node


class AutodiffMaxGradTestCase(ut.TestCase):
    jax.config.update("jax_enable_x64", True)

    ###################################
    # MAX GRAD AND MAX NEG GRAD TESTS #
    ###################################
    def test_heaviest_gradient_path_two_inputs(self):
        x_bf = bf.Parameter(4, name="x")
        y_bf = bf.Parameter(2, name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop()

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[1].item().name, "(* x y)")

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[0], 0.5)
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(/ x y)")

        # Check the positive max grad of inputly has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertEqual(y_bf.max_grad_of_output_wrt_node[0], 4.0)
        self.assertEqual(y_bf.max_grad_of_output_wrt_node[1].item().name, "(* x y)")

        # Check the negative max grad of inputly has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertEqual(y_bf.max_neg_grad_of_output_wrt_node[0], -1.0)
        self.assertEqual(y_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(/ x y)")

    def test_heaviest_gradient_path_multiple_paths_with_expected_positive_max_grad(
        self,
    ):
        x_bf = bf.Parameter(0, name="x")
        sinx_bf = bf.math.sin(x_bf)
        output = (sinx_bf + 1) + 2 * sinx_bf

        output.backprop()

        # Check the positive max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[1], sinx_bf)

        # Check the negative max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[1], sinx_bf)

        # Check the positive max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[1].item().name, "(* 2 (sin x))")

        # Check the negative max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(+ (sin x) 1)")

    def test_heaviest_gradient_path_multiple_paths_associativity_with_expected_positive_max_grad(
        self,
    ):
        x_bf = bf.Parameter(0, name="x")
        sinx_bf = bf.math.sin(x_bf)
        output = 2 * sinx_bf + (sinx_bf + 1)

        output.backprop()

        # Check the positive max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[1], sinx_bf)

        # Check the negative max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[1], sinx_bf)

        # Check the positive max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[1].item().name, "(* 2 (sin x))")

        # Check the negative max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(+ (sin x) 1)")

    def test_heaviest_gradient_path_multiple_paths_with_expected_positive_max_grad_duplicate_nodes(
        self,
    ):
        x_bf = bf.Parameter(0, name="x")
        output = (bf.math.sin(x_bf) + 1) + 2 * bf.math.sin(x_bf)

        output.backprop()

        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], 2.0)
        # Check the positive max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[1].item().name, "(sin x)")

        # Check the negative max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(sin x)")

    def test_heaviest_gradient_path_multiple_paths_with_expected_negative_max_grad(
        self,
    ):
        x_bf = bf.Parameter(jnp.pi, name="x")
        sinx_bf = bf.math.sin(x_bf)
        output = (sinx_bf + 1) + 2 * sinx_bf

        output.backprop()

        # Check the positive max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], -1.0)
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[1], sinx_bf)

        # Check the negative max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[0], -2.0)
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[1], sinx_bf)

        # Check the positive max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[1].item().name, "(* 2 (sin x))")

        # Check the negative max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(+ (sin x) 1)")

    def test_heaviest_gradient_path_multiple_paths_associativity_with_expected_negative_max_grad(
        self,
    ):
        x_bf = bf.Parameter(jnp.pi, name="x")
        sinx_bf = bf.math.sin(x_bf)
        output = 2 * sinx_bf + (sinx_bf + 1)

        output.backprop()

        # Check the positive max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[0], -1.0)
        self.assertEqual(x_bf.max_grad_of_output_wrt_node[1], sinx_bf)

        # Check the negative max grad of input x has expected value and parent node
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[0], -2.0)
        self.assertEqual(x_bf.max_neg_grad_of_output_wrt_node[1], sinx_bf)

        # Check the positive max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[0], 2.0)
        self.assertEqual(sinx_bf.max_grad_of_output_wrt_node[1].item().name, "(* 2 (sin x))")

        # Check the negative max grad of parent node of x (which is the sin x node) has expected value and parent node
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[0], 1.0)
        self.assertEqual(sinx_bf.max_neg_grad_of_output_wrt_node[1].item().name, "(+ (sin x) 1)")

    def test_heaviest_gradient_path_two_vector_inputs(self):
        x_bf = bf.Parameter(jnp.array([4.0, 12.0]), name="x")
        y_bf = bf.Parameter(jnp.array([2.0, 4.0]), name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(x_bf.max_grad_of_output_wrt_node[0], [2.0, 4.0]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in x_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(* x y)"])
        )

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(x_bf.max_neg_grad_of_output_wrt_node[0], [0.5, 0.25]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in x_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(/ x y)"])
        )

        # Check the positive max grad of inputly has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(y_bf.max_grad_of_output_wrt_node[0], [4.0, 12.0]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in y_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(* x y)"])
        )

        # Check the negative max grad of inputly has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(y_bf.max_neg_grad_of_output_wrt_node[0], [-1.0, -0.75]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in y_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(/ x y)"])
        )

    def test_heaviest_gradient_path_two_vector_inputs_with_differing_elementwise_max_grads(self):
        x_bf = bf.Parameter(jnp.array([4.0, -12.0]), name="x")
        y_bf = bf.Parameter(jnp.array([2.0, 4.0]), name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(x_bf.max_grad_of_output_wrt_node[0], [2.0, 4.0]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in x_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(* x y)"])
        )

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(x_bf.max_neg_grad_of_output_wrt_node[0], [0.5, 0.25]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in x_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(/ x y)"])
        )

        # Check the positive max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(y_bf.max_grad_of_output_wrt_node[0], [4.0, 0.75]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in y_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(/ x y)"])
        )

        # Check the negative max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(y_bf.max_neg_grad_of_output_wrt_node[0], [-1.0, -12.0]))
        self.assertTrue(
            jnp.array_equal([parent.name for parent in y_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(* x y)"])
        )

    def test_heaviest_gradient_path_two_batched_vector_inputs(self):
        x_bf = bf.Parameter(jnp.array([[4.0, 12.0], [4.0, -12.0]]), name="x")
        y_bf = bf.Parameter(jnp.array([[2.0, 4.0], [2.0, 4.0]]), name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(x_bf.max_grad_of_output_wrt_node[0], [[2.0, 4.0], [2.0, 4.0]]))
        self.assertTrue(
            jnp.array_equal(
                jnp.vectorize(lambda node: node.name)(x_bf.max_grad_of_output_wrt_node[1]),
                [["(* x y)", "(* x y)"], ["(* x y)", "(* x y)"]],
            )
        )

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(x_bf.max_neg_grad_of_output_wrt_node[0], [[0.5, 0.25], [0.5, 0.25]]))
        self.assertTrue(
            jnp.array_equal(
                jnp.vectorize(lambda node: node.name)(x_bf.max_neg_grad_of_output_wrt_node[1]),
                [["(/ x y)", "(/ x y)"], ["(/ x y)", "(/ x y)"]],
            )
        )

        # Check the positive max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(y_bf.max_grad_of_output_wrt_node[0], [[4.0, 12.0], [4.0, 0.75]]))
        self.assertTrue(
            jnp.array_equal(
                jnp.vectorize(lambda node: node.name)(y_bf.max_grad_of_output_wrt_node[1]),
                [["(* x y)", "(* x y)"], ["(* x y)", "(/ x y)"]],
            )
        )

        # Check the negative max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(jnp.array_equal(y_bf.max_neg_grad_of_output_wrt_node[0], [[-1.0, -0.75], [-1.0, -12.0]]))
        self.assertTrue(
            jnp.array_equal(
                jnp.vectorize(lambda node: node.name)(y_bf.max_neg_grad_of_output_wrt_node[1]),
                [["(/ x y)", "(/ x y)"], ["(/ x y)", "(* x y)"]],
            )
        )

    def test_heaviest_gradient_path_linear(self):
        x_bf = bf.Parameter(jnp.array([[3.0]]), name="x")
        y_bf = bf.Parameter(jnp.array([0]), name="y")

        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[4.0, 1.0]]))
        linear_bf.set_bias(jnp.array([0.0, 0.0]))

        output = linear_bf(x_bf)
        loss = bf.opt.cross_entropy_loss(output, y_bf)

        loss.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(
            jnp.allclose(x_bf.max_grad_of_output_wrt_node[0], [[4.00012339]])
        )  # this comes from matmul(outgrad, weights), as defined by the matmul derivative. So, [[1.0, 1.2339e-04]] * [[4, 1]] = 4 + 0.00012339.
        self.assertTrue(
            jnp.allclose(x_bf.max_neg_grad_of_output_wrt_node[0], [[-4.0]])
        )  # this comes from matmul(outgrad, weights), as defined by the matmul derivative. So, [[-1.0, 0.]] * [[4, 1]] = -4.
        self.assertTrue(
            x_bf.max_grad_of_output_wrt_node[0] * x_bf + x_bf.max_neg_grad_of_output_wrt_node[0] * x_bf == x_bf.grad
        )

        print("x max grad:", x_bf.max_grad_of_output_wrt_node)
        print("x max neg grad:", x_bf.max_neg_grad_of_output_wrt_node)
        print("x grad:", x_bf.grad)

        print("w max grad:", linear_bf.W.max_grad_of_output_wrt_node)
        print("w max neg grad:", linear_bf.W.max_neg_grad_of_output_wrt_node)
        print("w grad:", linear_bf.W.grad)


class AutodiffEntropyTestCase(ut.TestCase):
    #################
    # ENTROPY TESTS #
    #################
    def test_entropy_of_single_variable_in_two_paths_with_identical_gradients(self):
        x_bf = bf.Parameter(3, name="x")
        output = x_bf * 2 + x_bf * 2
        output.backprop(verbose=True)

        self.assertTrue(x_bf.compute_entropy().val == jnp.log(2))

    def test_entropy_when_input_has_0_val(self):
        x_bf = bf.Parameter(0, name="x")
        output = x_bf * 2 + x_bf * 2
        output.backprop(verbose=True)

        self.assertTrue(x_bf.compute_entropy().val == jnp.log(2))

    def test_entropy_of_single_variable_in_two_paths_with_opposite_gradients(self):
        x_bf = bf.Parameter(3, name="x")
        output = x_bf * 4 + x_bf * -2
        output.backprop(verbose=True)

        print("x_bf.compute_entropy():", x_bf.compute_entropy())
        print("x_bf grad:", x_bf.grad)

        self.assertTrue(x_bf.compute_entropy().val == ((-4 * jnp.log(4) - 2 * jnp.log(2)) / 6 + jnp.log(6)))

    def test_entropy_of_single_variable_in_two_paths_with_identical_but_opposite_gradients(self):
        x_bf = bf.Parameter(3, name="x")
        output = x_bf * 2 + x_bf * -2
        output.backprop(verbose=True)

        print("x_bf.compute_entropy():", x_bf.compute_entropy())
        print("x_bf grad:", x_bf.grad)

        self.assertTrue(x_bf.compute_entropy().val == jnp.log(2))

    def test_entropy_of_vector_variable_in_two_paths_with_identical_but_opposite_gradients(self):
        x_bf = bf.Parameter(jnp.array([3.0, 5.0, 0.0]), name="x")
        output = x_bf * 2 + x_bf * -2
        output.backprop(verbose=True)

        print("x_bf.compute_entropy():", x_bf.compute_entropy())
        print("x_bf grad:", x_bf.grad)

        self.assertTrue(jnp.array_equal(x_bf.compute_entropy().val, jnp.array([jnp.log(2), jnp.log(2), jnp.log(2)])))

    def test_entropy_of_vector_variable_in_two_paths_with_input_dependent_gradients(self):
        # Tests 3 things - (1) this works vectorized, (2) this works when you pass in different elemnts in each vector, and (3) the negative gradient gets abs val'd and works too.
        x_bf = bf.Parameter(jnp.array([3.0, 5.0, -3.0]), name="x")
        output = x_bf * 3 + x_bf**2
        output.backprop(verbose=True)

        print("x_bf.compute_entropy():", x_bf.compute_entropy())
        print("x_bf grad:", x_bf.grad)

        print((-2 * 3 * jnp.log(2 * 3) - 3 * jnp.log(3)) / (2 * 3 + 3) + jnp.log(2 * 3 + 3))
        self.assertTrue(
            jnp.array_equal(
                x_bf.compute_entropy().val,
                jnp.array(
                    [
                        (-2 * 3 * jnp.log(2 * 3) - 3 * jnp.log(3)) / (2 * 3 + 3) + jnp.log(2 * 3 + 3),
                        (-2 * 5 * jnp.log(2 * 5) - 3 * jnp.log(3)) / (2 * 5 + 3) + jnp.log(2 * 5 + 3),
                        (-2 * 3 * jnp.log(2 * 3) - 3 * jnp.log(3)) / (2 * 3 + 3) + jnp.log(2 * 3 + 3),
                    ]
                ),
            )
        )

    def test_entropy_of_vector_variable_in_three_paths_with_x_times_x(self):
        x_bf = bf.Parameter(jnp.array([3.0, 5.0, -3.0]), name="x")
        output = x_bf * 3 + x_bf * x_bf
        output.backprop(verbose=True)

        print("x_bf.compute_entropy():", x_bf.compute_entropy())
        print("x_bf grad:", x_bf.grad)

        self.assertTrue(
            jnp.array_equal(
                x_bf.compute_entropy().val,
                jnp.array(
                    [
                        (-2 * 3 * jnp.log(3) - 3 * jnp.log(3)) / (2 * 3 + 3) + jnp.log(2 * 3 + 3),
                        (-2 * 5 * jnp.log(5) - 3 * jnp.log(3)) / (2 * 5 + 3) + jnp.log(2 * 5 + 3),
                        (-2 * 3 * jnp.log(3) - 3 * jnp.log(3)) / (2 * 3 + 3) + jnp.log(2 * 3 + 3),
                    ]
                ),
            )
        )

    def test_entropy_reduce_logsumexp(self):
        # Note - there's no negs here, but assuming negs work because it's been tested for smaller component functions.
        output_bf = bf.Parameter(jnp.array([[2.0, 1.0]]), name="output")
        reduced_bf = bf.func.reduce_logsumexp(output_bf, axis=-1)
        reduced_bf.backprop(verbose=True)

        self.assertTrue(
            jnp.allclose(
                output_bf.grad, jnp.array([[1 / (1 + 1 / jnp.e) * jnp.exp(0), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]])
            )
        )
        self.assertTrue(
            jnp.allclose(
                output_bf.abs_val_grad,
                jnp.array([[2 + 1 / (1 + 1 / jnp.e) * jnp.exp(0), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                output_bf.entropy_wrt_output,
                jnp.array(
                    [
                        [
                            -jnp.array(1 / (1 + 1 / jnp.e)) * jnp.log(jnp.array(1 / (1 + 1 / jnp.e)))
                            + jnp.sum(
                                -jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]])
                                * jnp.log(jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]]))
                            ),
                            (
                                -jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]])
                                * jnp.log(jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]]))
                            )[0, 1],
                        ]
                    ]
                ),
            )
        )

    def test_entropy_cross_entropy_loss_until_reduce_log_sum_exp(self):
        output_bf = bf.Parameter(jnp.array([[2.0, 1.0]]), name="output")
        label_bf = bf.Parameter(jnp.array([0]), name="label")
        loss = bf.opt.loss.cross_entropy_loss(output_bf, label_bf, reduction="sum")
        loss.backprop(verbose=True)

        self.assertTrue(jnp.allclose(output_bf.grad, jnp.array([[-0.26894142, 0.26894142]])))
        self.assertTrue(jnp.allclose(output_bf.abs_val_grad, jnp.array([[3.73105858, 0.26894142]])))

        # This is actually the same because the entropy added from the reduce_logsumexp -> CEL is 0,
        # and the direct contribution of the arc from output_bf -> CEL is 0.
        self.assertTrue(
            jnp.allclose(
                output_bf.entropy_wrt_output,
                jnp.array(
                    [
                        [
                            -jnp.array(1 / (1 + 1 / jnp.e)) * jnp.log(jnp.array(1 / (1 + 1 / jnp.e)))
                            + jnp.sum(
                                -jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]])
                                * jnp.log(jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]]))
                            ),
                            (
                                -jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]])
                                * jnp.log(jnp.array([[1 / (1 + 1 / jnp.e), 1 / (1 + 1 / jnp.e) * jnp.exp(-1)]]))
                            )[0, 1],
                        ]
                    ]
                ),
            )
        )

        # This is slightly different now because the total (abs_val) gradient does change,
        # so even though the unnormalized entropy is the same, the actual entropy changes.
        self.assertTrue(jnp.allclose(output_bf.compute_entropy().val, jnp.array([[1.53411441e00, 4.44089210e-16]])))

    def test_entropy_linear(self):
        x_bf = bf.Parameter(jnp.array([[3.0]]), name="x")

        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[2.0, 1.0]]))
        linear_bf.set_bias(jnp.array([0.0, 0.0]))

        output = linear_bf(x_bf)
        output.backprop(verbose=True)

        # There's two paths x can take to the end - one with weight 2, 1 with weight 1, so it's just the entropy of that distribution.
        self.assertTrue(jnp.allclose(x_bf.compute_entropy().val, jnp.array([[ss.entropy([2 / 3, 1 / 3])]])))

    def test_entropy_linear_neg_weights(self):
        x_bf = bf.Parameter(jnp.array([[3.0]]), name="x")

        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[-2.0, 1.0]]))
        linear_bf.set_bias(jnp.array([0.0, 0.0]))

        output = linear_bf(x_bf)
        output.backprop(verbose=True)

        # There's two paths x can take to the end - one with weight 2, 1 with weight 1, so it's just the entropy of that distribution.
        self.assertTrue(jnp.allclose(x_bf.compute_entropy().val, jnp.array([[ss.entropy([2 / 3, 1 / 3])]])))

    def test_entropy_linear_with_bias(self):
        x_bf = bf.Parameter(jnp.array([[3.0]]), name="x")

        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[-2.0, 1.0]]))
        linear_bf.set_bias(jnp.array([4.0, 5.0]))

        output = linear_bf(x_bf)
        output.backprop(verbose=True)

        # Even with bias, there's two paths x can take to the end -
        # and the decision point has one with weight 2, 1 with weight 1, so it's just the entropy of that distribution.
        self.assertTrue(jnp.allclose(x_bf.compute_entropy().val, jnp.array([[ss.entropy([2 / 3, 1 / 3])]])))

    def test_entropy_no_nans(self):
        """Test that any NaNs computed in the entropy_wrt_output because of 0log0 are set to 0."""
        x_bf = bf.Parameter(jnp.array([-1, 0, 1]), name="x")
        output = bf.relu(x_bf)
        output.backprop(verbose=True)

        # The 0th and 1st elements would otherwise be NaNs if computing entropy did not set 0log0 = 0.
        assert jnp.array_equal(x_bf.entropy_wrt_output, jnp.array([0, 0, 0]))


class BruteForceTestCase(ut.TestCase):
    def test_get_all_paths_base_case(self):
        x_bf = bf.Parameter(4, name="x")
        y_bf = bf.Parameter(2, name="y")
        x_bf.backprop()

        self.assertTrue(len(get_all_paths_to_node(x_bf)) == 1)
        self.assertTrue(len(get_all_paths_from_src_to_target_node(x_bf, x_bf)) == 1)
        self.assertTrue(len(get_all_paths_from_src_to_target_node(x_bf, y_bf)) == 0)

    def test_get_all_paths_simple(self):
        x_bf = bf.Parameter(4, name="x")
        y_bf = bf.Parameter(2, name="y")
        output = x_bf * y_bf + x_bf / y_bf
        output.backprop()

        self.assertTrue(len(get_all_paths_to_node(output)) == 4)
        self.assertTrue(len(get_all_paths_from_src_to_target_node(output, x_bf)) == 2)

    def test_get_all_paths_complex(self):
        x_bf = bf.Parameter(jnp.array([4, 2]), name="x")
        y_bf = bf.Parameter(jnp.array([2, 3]), name="y")
        z_bf = bf.Parameter(jnp.array([2, 3]), name="z")
        output = x_bf * y_bf * z_bf + x_bf / y_bf / z_bf - y_bf * x_bf / z_bf
        output.backprop()

        self.assertEqual(len(get_all_paths_to_node(output)), 9)
        self.assertEqual(len(get_all_paths_from_src_to_target_node(output, x_bf)), 3)

        # These are weird cases where y and z have the same value, so they're considered the same starting node, lol
        self.assertEqual(len(get_all_paths_from_src_to_target_node(output, y_bf)), 6)
        self.assertEqual(len(get_all_paths_from_src_to_target_node(output, z_bf)), 6)


class RegularizationTestCase(ut.TestCase):
    ########################
    # REGULARIZATION TESTS #
    ########################
    def test_l1_regularization(self):
        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[4.0, 1.0]]))
        linear_bf.set_bias(jnp.array([2.0, 0.0]))

        regularized = regularize(linear_bf, l1_weight=1, l2_weight=0)

        assert regularized.val == 7.0
        regularized.backprop(verbose=True)

        self.assertTrue(jnp.allclose(linear_bf.W.grad, jnp.array([[1.0, 1.0]])))
        self.assertTrue(jnp.allclose(linear_bf.b.grad, jnp.array([[1.0, 1.0]])))

    def test_l2_regularization(self):
        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[4.0, 1.0]]))
        linear_bf.set_bias(jnp.array([2.0, 0.0]))

        regularized = regularize(linear_bf, l1_weight=0, l2_weight=1)

        assert regularized.val == 21.0
        regularized.backprop(verbose=True, values_to_compute=["grad", "abs_val_grad", "entropy"])

        self.assertTrue(jnp.allclose(linear_bf.W.grad, jnp.array([[8.0, 2.0]])))
        self.assertTrue(jnp.allclose(linear_bf.b.grad, jnp.array([[4.0, 0.0]])))

    def test_l2_half_weighted_regularization(self):
        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(jnp.array([[4.0, 1.0]]))
        linear_bf.set_bias(jnp.array([2.0, 0.0]))

        regularized = regularize(linear_bf, l1_weight=0, l2_weight=0.5)

        assert regularized.val == 10.5
        regularized.backprop(verbose=True)

        self.assertTrue(jnp.allclose(linear_bf.W.grad, jnp.array([[4.0, 1.0]])))
        self.assertTrue(jnp.allclose(linear_bf.b.grad, jnp.array([[2.0, 0.0]])))
