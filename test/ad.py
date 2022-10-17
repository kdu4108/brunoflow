import brunoflow as bf
import numpy as np
import unittest as ut


class AutodiffMaxGradTestCase(ut.TestCase):
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
        x_bf = bf.Parameter(np.pi, name="x")
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
        x_bf = bf.Parameter(np.pi, name="x")
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
        x_bf = bf.Parameter(np.array([4.0, 12.0]), name="x")
        y_bf = bf.Parameter(np.array([2.0, 4.0]), name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(x_bf.max_grad_of_output_wrt_node[0], [2.0, 4.0]))
        self.assertTrue(
            np.array_equal([parent.name for parent in x_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(* x y)"])
        )

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(x_bf.max_neg_grad_of_output_wrt_node[0], [0.5, 0.25]))
        self.assertTrue(
            np.array_equal([parent.name for parent in x_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(/ x y)"])
        )

        # Check the positive max grad of inputly has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(y_bf.max_grad_of_output_wrt_node[0], [4.0, 12.0]))
        self.assertTrue(
            np.array_equal([parent.name for parent in y_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(* x y)"])
        )

        # Check the negative max grad of inputly has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(y_bf.max_neg_grad_of_output_wrt_node[0], [-1.0, -0.75]))
        self.assertTrue(
            np.array_equal([parent.name for parent in y_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(/ x y)"])
        )

    def test_heaviest_gradient_path_two_vector_inputs_with_differing_elementwise_max_grads(self):
        x_bf = bf.Parameter(np.array([4.0, -12.0]), name="x")
        y_bf = bf.Parameter(np.array([2.0, 4.0]), name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(x_bf.max_grad_of_output_wrt_node[0], [2.0, 4.0]))
        self.assertTrue(
            np.array_equal([parent.name for parent in x_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(* x y)"])
        )

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(x_bf.max_neg_grad_of_output_wrt_node[0], [0.5, 0.25]))
        self.assertTrue(
            np.array_equal([parent.name for parent in x_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(/ x y)"])
        )

        # Check the positive max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(y_bf.max_grad_of_output_wrt_node[0], [4.0, 0.75]))
        self.assertTrue(
            np.array_equal([parent.name for parent in y_bf.max_grad_of_output_wrt_node[1]], ["(* x y)", "(/ x y)"])
        )

        # Check the negative max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(y_bf.max_neg_grad_of_output_wrt_node[0], [-1.0, -12.0]))
        self.assertTrue(
            np.array_equal([parent.name for parent in y_bf.max_neg_grad_of_output_wrt_node[1]], ["(/ x y)", "(* x y)"])
        )

    def test_heaviest_gradient_path_two_batched_vector_inputs(self):
        x_bf = bf.Parameter(np.array([[4.0, 12.0], [4.0, -12.0]]), name="x")
        y_bf = bf.Parameter(np.array([[2.0, 4.0], [2.0, 4.0]]), name="y")
        output = x_bf * y_bf + x_bf / y_bf

        output.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(x_bf.max_grad_of_output_wrt_node[0], [[2.0, 4.0], [2.0, 4.0]]))
        self.assertTrue(
            np.array_equal(
                np.vectorize(lambda node: node.name)(x_bf.max_grad_of_output_wrt_node[1]),
                [["(* x y)", "(* x y)"], ["(* x y)", "(* x y)"]],
            )
        )

        # Check the negative max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(x_bf.max_neg_grad_of_output_wrt_node[0], [[0.5, 0.25], [0.5, 0.25]]))
        self.assertTrue(
            np.array_equal(
                np.vectorize(lambda node: node.name)(x_bf.max_neg_grad_of_output_wrt_node[1]),
                [["(/ x y)", "(/ x y)"], ["(/ x y)", "(/ x y)"]],
            )
        )

        # Check the positive max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(y_bf.max_grad_of_output_wrt_node[0], [[4.0, 12.0], [4.0, 0.75]]))
        self.assertTrue(
            np.array_equal(
                np.vectorize(lambda node: node.name)(y_bf.max_grad_of_output_wrt_node[1]),
                [["(* x y)", "(* x y)"], ["(* x y)", "(/ x y)"]],
            )
        )

        # Check the negative max grad of input y has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(np.array_equal(y_bf.max_neg_grad_of_output_wrt_node[0], [[-1.0, -0.75], [-1.0, -12.0]]))
        self.assertTrue(
            np.array_equal(
                np.vectorize(lambda node: node.name)(y_bf.max_neg_grad_of_output_wrt_node[1]),
                [["(/ x y)", "(/ x y)"], ["(/ x y)", "(* x y)"]],
            )
        )

    def test_heaviest_gradient_path_linear(self):
        x_bf = bf.Parameter(np.array([[3.0]]), name="x")
        y_bf = bf.Parameter(np.array([0]), name="y")

        linear_bf = bf.net.Linear(1, 2)
        linear_bf.set_weights(np.array([[4.0, 1.0]]))
        linear_bf.set_bias(np.array([0.0, 0.0]))

        output = linear_bf(x_bf)
        loss = bf.opt.cross_entropy_loss(output, y_bf)

        loss.backprop(verbose=True)

        # Check the positive max grad of input x has expected value and parent node (where parent in this case means "closer in the graph to the output")
        self.assertTrue(
            np.allclose(x_bf.max_grad_of_output_wrt_node[0], [[4.00012339]])
        )  # this comes from matmul(outgrad, weights), as defined by the matmul derivative. So, [[1.0, 1.2339e-04]] * [[4, 1]] = 4 + 0.00012339.
        self.assertTrue(
            np.allclose(x_bf.max_neg_grad_of_output_wrt_node[0], [[-4.0]])
        )  # this comes from matmul(outgrad, weights), as defined by the matmul derivative. So, [[-1.0, 0.]] * [[4, 1]] = -4.
        self.assertTrue(
            x_bf.max_grad_of_output_wrt_node[0] * x_bf + x_bf.max_neg_grad_of_output_wrt_node[0] * x_bf == x_bf.grad
        )
        print(x_bf.max_grad_of_output_wrt_node[0] * x_bf)

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

        self.assertTrue(x_bf.compute_entropy() == np.log(2))
