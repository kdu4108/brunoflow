import brunoflow as bf
import numpy as np
import unittest as ut


class AutodiffTestCase(ut.TestCase):
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
