from __future__ import annotations
from colorama import Style
import graphviz
import numpy as np
import numpy.typing as npt
from typing import Any, List, Tuple, Union

import brunoflow as bf


class Node:
    """
    Represents one node in an autodiff computation graph.

    Each Node stores a value and a gradient.
    Nodes are created by calling autodiff functions (the forward pass)
    The gradient of each Node is computed during the backward pass.

    Tensorflow analogue:
        tf.Tensor

    PyTorch analogue:
        torch.Tensor

    Attributes:
        val (numpy.ndarray | float): The value at this point in the forward pass over the
            computation graph
        grad (numpy.ndarray | float): The gradient at this point in the backward pass over
            the computation graph
        inputs (list): The inputs to the function which produced this Node as output
            (empty for leaf Nodes). These may be Nodes themselves, np.ndarrays, or other types
        backward_func: A function which takes self.grad and computes gradients for each
            of self.inputs (None for leaf Nodes)
        shape (tuple): The same as numpy.ndarray.shape
        size (int): The same as numpy.ndarray.size
        ndim (int): The same as numpy.ndarray.ndim
    """

    def __init__(
        self,
        val: Union[float, np.ndarray],
        backward_func=None,
        inputs: List[Union[Node, np.ndarray, Any]] = [],
        name: str = None,
    ):
        self.val = val
        self.grad: Union[np.ndarray, np.float64] = np.zeros_like(val, dtype=np.float64)
        self.max_grad_of_output_wrt_node: Tuple[Union[np.ndarray, np.float64], Union[npt.NDArray[Node], Node]] = (
            np.full_like(val, fill_value=-np.inf, dtype=np.float64),
            np.empty_like(val, dtype=type(None)),
        )
        self.max_neg_grad_of_output_wrt_node: Tuple[Union[np.ndarray, np.float64], Union[npt.NDArray[Node], Node]] = (
            np.full_like(val, fill_value=np.inf, dtype=np.float64),
            np.empty_like(val, dtype=type(None)),
        )
        self.entropy_wrt_output: Union[np.ndarray, np.float64] = np.zeros_like(val, dtype=np.float64)
        self.abs_val_grad: Union[np.ndarray, np.float64] = np.zeros_like(val, dtype=np.float64)
        self.backward_func = backward_func
        self.inputs = inputs
        self.name = name
        self.__num_uses = 0
        self.graph = graphviz.Digraph("computation_graph", filename="computation_graph.gv")

    def set_name(self, new_name: str):
        self.name = new_name

    @property
    def shape(self):
        return self.val.shape

    @property
    def size(self):
        return self.val.size

    @property
    def ndim(self):
        return self.val.ndim

    def __str__(self):
        return f"node(name: {self.name}, val: {self.val}, grad: {self.grad})"

    def __repr__(self):
        return str(self)

    def __compute_num_uses(self):
        """
        Starting at this node, recursively compute how many times each Node is used as an
            input by some other Node in the graph.
        This is used during backpropagation to determine when all the gradients for the
            Node have been accmulated and the backward pass can move on to its inputs
        """
        self.__num_uses += 1
        if self.__num_uses == 1:
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__compute_num_uses()

    def backprop(self, values_to_compute=("grad", "max_grad", "abs_val_grad", "entropy"), verbose=False):
        """
        Initiate a backward pass starting from this Node.
        Computes gradients from every Node reachable from this Node, back to the leaves.
        """
        # print(f"Entering backprop on node {self.name}:", self)
        if isinstance(self.grad, np.ndarray):
            self.grad.fill(1.0)
            self.max_grad_of_output_wrt_node[0].fill(1.0)
            self.max_neg_grad_of_output_wrt_node[0].fill(1.0)
            self.entropy_wrt_output.fill(0.0)
            self.abs_val_grad.fill(1.0)
        else:
            # If the grad is a float rather than an np.ndarray
            self.grad = 1.0
            self.max_grad_of_output_wrt_node = (1.0, None)
            self.max_neg_grad_of_output_wrt_node = (1.0, None)
            self.entropy_wrt_output = 0.0
            self.abs_val_grad = 1.0
        if verbose:
            print("Starting backprop:")
            print(
                self.name,
                ", num_uses:",
                self.__num_uses,
                ", max_grad:",
                self.max_grad_of_output_wrt_node[0],
                ", type max_grad:",
                type(self.max_grad_of_output_wrt_node[0]),
                ", grad:",
                self.grad,
                ", type grad:",
                type(self.grad),
            )

        self.__compute_num_uses()
        self.__backprop(values_to_compute=values_to_compute, verbose=verbose)

    def _compute_and_accumulate_grads_for_inputs(self, input_vals, backward_func, verbose=False):
        # The 'adjoint' is the partial derivative of the final node in the graph
        #   (typically the loss) w.r.t. the value at this Node.
        adjoints = backward_func(self.val, self.grad, *input_vals)

        assert len(input_vals) == len(adjoints)
        # Accumulate these adjoints into the gradients for this Node's inputs
        for i in range(len(adjoints)):
            adj = adjoints[i]

            if isinstance(self.inputs[i], Node):
                assert (
                    self.inputs[i].grad.shape
                    == self.inputs[i].max_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].max_neg_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].entropy_wrt_output.shape
                    == self.inputs[i].abs_val_grad.shape
                )
                # An adjoint may need to be 'reshaped' before it can be accumulated if the forward operation used broadcasting.
                adjoint_gradient = reshape_adjoint(adj, self.inputs[i].grad.shape)

                # ACCUMULATE GRADIENT BY SUMMING
                prev_grad = np.copy(self.inputs[i].grad)
                self.inputs[i].grad += adjoint_gradient
                if verbose:
                    print(
                        f"    {Style.RESET_ALL}Updating grad of node {self.inputs[i].name}:\n     from {prev_grad} \n       to {self.inputs[i].grad}."
                    )
                del prev_grad

    def _compute_and_accumulate_max_pos_and_neg_grads_for_inputs(self, input_vals, backward_func, verbose=False):
        assert (self.max_grad_of_output_wrt_node[0] >= self.max_neg_grad_of_output_wrt_node[0]).all()

        # Multiply the max pos grad of self w.r.t. output by the local derivative for each input val
        adjoints_max_grad = backward_func(self.val, self.max_grad_of_output_wrt_node[0], *input_vals)

        # Multiply the max neg grad of self w.r.t. output by the local derivative for each input val
        adjoints_max_neg_grad = backward_func(self.val, self.max_neg_grad_of_output_wrt_node[0], *input_vals)

        assert len(input_vals) == len(adjoints_max_grad) == len(adjoints_max_neg_grad)
        # Accumulate these adjoints into the gradients for this Node's inputs
        for i in range(len(adjoints_max_grad)):
            adj_max_grad = adjoints_max_grad[i]
            adj_max_neg_grad = adjoints_max_neg_grad[i]

            if isinstance(self.inputs[i], Node):
                assert (
                    self.inputs[i].grad.shape
                    == self.inputs[i].max_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].max_neg_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].entropy_wrt_output.shape
                    == self.inputs[i].abs_val_grad.shape
                )
                # An adjoint may need to be 'reshaped' before it can be accumulated if the forward operation used broadcasting.
                adj_max_grad = reshape_adjoint(adj_max_grad, self.inputs[i].grad.shape)
                adj_max_neg_grad = reshape_adjoint(adj_max_neg_grad, self.inputs[i].grad.shape)

                # If the max pos grad is less than the max neg grad, that means there was some sign flipping, and we should reverse which is most positive and most negative. Use np.maximum and np.minimum to do this element-wise.
                adj_max_grad, adj_max_neg_grad = np.maximum(adj_max_grad, adj_max_neg_grad), np.minimum(
                    adj_max_grad, adj_max_neg_grad
                )

                # ACCUMULATE MAX POS GRAD BY USING THE MAX OPERATOR
                # Replace the current (max grad value, parent) of the ith input node with (max grad value of self, self) for all elements in the ith input node that have higher max grad values in self.
                inds_to_replace_max_grad = adj_max_grad > self.inputs[i].max_grad_of_output_wrt_node[0]
                prev_vals, prev_parents = np.copy(self.inputs[i].max_grad_of_output_wrt_node[0]), np.copy(
                    self.inputs[i].max_grad_of_output_wrt_node[1]
                )
                self.inputs[i].max_grad_of_output_wrt_node[0][inds_to_replace_max_grad] = adj_max_grad[
                    inds_to_replace_max_grad
                ]
                self.inputs[i].max_grad_of_output_wrt_node[1][inds_to_replace_max_grad] = self
                if verbose and not np.array_equal(
                    (prev_vals, prev_parents), self.inputs[i].max_grad_of_output_wrt_node
                ):
                    print(
                        f"    Replacing max_grad of node {self.inputs[i].name}:\n     from {(prev_vals, prev_parents)} \n       to {self.inputs[i].max_grad_of_output_wrt_node}."
                    )
                del prev_vals, prev_parents

                # ACCUMULATE MAX NEG GRAD BY USING THE MIN OPERATOR
                # Replace the current (max neg grad value, parent) of the ith input node with (max neg grad value of self, self) for all elements in the ith input node that have more negative max neg grad values in self.
                inds_to_replace_max_neg_grad = adj_max_neg_grad < self.inputs[i].max_neg_grad_of_output_wrt_node[0]
                prev_vals, prev_parents = np.copy(self.inputs[i].max_neg_grad_of_output_wrt_node[0]), np.copy(
                    self.inputs[i].max_neg_grad_of_output_wrt_node[1]
                )
                self.inputs[i].max_neg_grad_of_output_wrt_node[0][inds_to_replace_max_neg_grad] = adj_max_neg_grad[
                    inds_to_replace_max_neg_grad
                ]
                self.inputs[i].max_neg_grad_of_output_wrt_node[1][inds_to_replace_max_neg_grad] = self
                if verbose and not np.array_equal(
                    (prev_vals, prev_parents), self.inputs[i].max_neg_grad_of_output_wrt_node
                ):
                    print(
                        f"    Replacing max_neg_grad of node {self.inputs[i].name}:\n     from {(prev_vals, prev_parents)} \n       to {self.inputs[i].max_neg_grad_of_output_wrt_node}."
                    )
                del prev_vals, prev_parents

    def _compute_and_accumulate_abs_val_grads_for_inputs(self, input_vals, backward_func, verbose=False):
        # Required for running the entropy semiring properly, compute the abs value of the gradient
        adjoints_abs_val_grad = backward_func(
            self.val,
            dict(out_abs_val_grad=self.abs_val_grad),
            *input_vals,
            product_fn=(lambda l_adj, out_abs_val_grad_dict: np.abs(l_adj) * out_abs_val_grad_dict["out_abs_val_grad"]),
        )

        assert len(input_vals) == len(adjoints_abs_val_grad)
        # Accumulate these adjoints into the gradients for this Node's inputs
        for i in range(len(adjoints_abs_val_grad)):
            adj_abs_val_grad = adjoints_abs_val_grad[i]
            if isinstance(self.inputs[i], Node):
                assert (
                    self.inputs[i].grad.shape
                    == self.inputs[i].max_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].max_neg_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].entropy_wrt_output.shape
                    == self.inputs[i].abs_val_grad.shape
                )
                assert (
                    adj_abs_val_grad >= 0
                ).all(), f"adj_abs_val_grad for output node w.r.t. input node {self.inputs[i]} via node {self} is {adj_abs_val_grad}. This number must be positive instead, so probably the backward func from {self} to {self.inputs[i]} is funky. Check the always-positive invariant holds!"

                # An adjoint may need to be 'reshaped' before it can be accumulated if the forward operation used broadcasting.
                adj_abs_val_grad = reshape_adjoint(adj_abs_val_grad, self.inputs[i].grad.shape)

                # ACCUMULATE ABS_VAL_GRAD BY USING THE SUM OPERATOR
                prev_abs_val_grad = np.copy(self.inputs[i].abs_val_grad)
                self.inputs[i].abs_val_grad += adj_abs_val_grad
                if verbose:
                    print(
                        f"    {Style.RESET_ALL}Updating abs_val_grad of node {self.inputs[i].name}:\n     from {prev_abs_val_grad} \n       to {self.inputs[i].abs_val_grad}."
                    )
                del prev_abs_val_grad

    def _compute_and_accumulate_entropy_for_inputs(self, input_vals, backward_func, verbose=False):
        # Semi-ring product (out_grad, -out_entropy) and (adj_grad, - adj_grad * log(adj_grad)) to get adjoint entropy (where adj_grad is the local derivative of self wrt each adjoint input)
        # print("backward_func:", backward_func.__name__)
        adjoints_entropy = backward_func(
            self.val,
            dict(out_abs_val_grad=self.abs_val_grad, out_entropy=self.entropy_wrt_output),
            *input_vals,
            product_fn=(
                lambda l_adj, out_grad_and_entropy_dict: np.abs(l_adj) * out_grad_and_entropy_dict["out_entropy"]
                + np.nan_to_num(
                    -np.abs(l_adj) * np.log(np.abs(l_adj)) * out_grad_and_entropy_dict["out_abs_val_grad"],
                    copy=False,
                    nan=0.0,
                )
            ),
        )  # TODO: this is broken for the inv function, fix that.

        assert len(input_vals) == len(adjoints_entropy)
        # Accumulate these adjoints into the gradients for this Node's inputs
        for i in range(len(adjoints_entropy)):
            adj_entropy = adjoints_entropy[i]

            if isinstance(self.inputs[i], Node):
                assert (
                    self.inputs[i].grad.shape
                    == self.inputs[i].max_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].max_neg_grad_of_output_wrt_node[0].shape
                    == self.inputs[i].entropy_wrt_output.shape
                    == self.inputs[i].abs_val_grad.shape
                )

                # An adjoint may need to be 'reshaped' before it can be accumulated if the forward operation used broadcasting.
                adj_entropy = reshape_adjoint(adj_entropy, self.inputs[i].grad.shape)

                # ACCUMULATE ENTROPY BY USING THE SUM OPERATOR
                prev_entropy = np.copy(self.inputs[i].entropy_wrt_output)
                self.inputs[i].entropy_wrt_output += adj_entropy
                if verbose:
                    print(
                        f"    {Style.RESET_ALL}Updating unnormalized entropy of node {self.inputs[i].name}:\n     from {prev_entropy} \n       to {self.inputs[i].entropy_wrt_output}."
                    )
                    print(
                        f"    {Style.RESET_ALL}Updating actual entropy of node {self.inputs[i].name}:\n     to {self.inputs[i].compute_entropy()}."
                    )
                del prev_entropy

    def __backprop(self, values_to_compute=("grad", "max_grad", "abs_val_grad", "entropy"), verbose=False):
        """
        Recursive helper function for self.backprop()
        Assumes that self.__compute_num_uses() has been called in advance
        """
        assert (self.abs_val_grad >= 0).all()
        assert ("abs_val_grad" in values_to_compute and "entropy" in values_to_compute) or (
            "abs_val_grad" not in values_to_compute and "entropy" not in values_to_compute
        )

        if verbose:
            print(
                f"{Style.RESET_ALL}\nCalling __backprop on node {self.name}",
                ", num_uses:",
                self.__num_uses,
                ", max_grad:",
                self.max_grad_of_output_wrt_node[0],
                ", grad:",
                self.grad,
                ", abs_val_grad:",
                self.abs_val_grad,
                ", entropy_wrt_output:",
                self.entropy_wrt_output,
            )
        # Record that backprop has reached this Node one more time
        self.__num_uses -= 1
        # If backprop has reached this Node as many times as it has been used as an input,
        #   then all of its gradients have been accmulated and backprop can continue up the graph.
        #   It will ONLY continue up the graph once all the grads have been accumulated
        if self.__num_uses == 0:
            backward_func = self.backward_func
            # The backward_func is the derivative of self with respect to its inputs
            if backward_func:
                input_vals = [inp.val if isinstance(inp, Node) else inp for inp in self.inputs]
                if "grad" in values_to_compute:
                    self._compute_and_accumulate_grads_for_inputs(
                        input_vals=input_vals, backward_func=backward_func, verbose=verbose
                    )
                if "max_grad" in values_to_compute:
                    self._compute_and_accumulate_max_pos_and_neg_grads_for_inputs(
                        input_vals=input_vals, backward_func=backward_func, verbose=verbose
                    )
                if "abs_val_grad" in values_to_compute:
                    self._compute_and_accumulate_abs_val_grads_for_inputs(
                        input_vals=input_vals, backward_func=backward_func, verbose=verbose
                    )
                if "entropy" in values_to_compute:
                    self._compute_and_accumulate_entropy_for_inputs(
                        input_vals=input_vals, backward_func=backward_func, verbose=verbose
                    )

            # Continue recursively backpropagating
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__backprop(values_to_compute=values_to_compute, verbose=verbose)

    def zero_gradients(self):
        """
        Zero out the gradients for all Nodes reachable from this Node (including this Node).
        Typically called before performing an optimization step, to ensure that gradients
            do not spuriously accumulate across iterations of optimization.
        """
        self.__compute_num_uses()
        self.__zero_gradients()

    def __zero_gradients(self):
        """
        Recursive helper for self.zero_gradients
        Assumes that self.__compute_num_uses() has been called in advance
        """
        # We use self.__num_uses to ensure that we traverse each Node in the graph once.
        # Since all we're doing is zeroing out gradients, it technically wouldn't be
        #    incorrect to traverse them multiple times, but it could be a lot less efficient.
        self.__num_uses -= 1
        if self.__num_uses == 0:
            if isinstance(self.grad, np.ndarray):
                self.grad.fill(0.0)
                # Careful - ndarray.fill() when type(ndarray) is int results in this int overflow thing instead of inf.
                self.max_grad_of_output_wrt_node = (
                    np.full_like(self.max_grad_of_output_wrt_node[0], fill_value=-np.inf, dtype=np.float64),
                    np.empty_like(self.max_grad_of_output_wrt_node[1], dtype=type(None)),
                )
                self.max_neg_grad_of_output_wrt_node = (
                    np.full_like(self.max_neg_grad_of_output_wrt_node[0], fill_value=np.inf, dtype=np.float64),
                    np.empty_like(self.max_neg_grad_of_output_wrt_node[1], dtype=type(None)),
                )
                self.entropy_wrt_output.fill(0.0)
                self.abs_val_grad.fill(0.0)
            else:
                self.grad = 0.0
                self.max_grad_of_output_wrt_node = (0.0, None)
                self.max_neg_grad_of_output_wrt_node = (0.0, None)
                self.entropy_wrt_output = 0.0
                self.abs_val_grad = 0.0
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__zero_gradients()

    def compute_entropy(self):
        return Node(self.entropy_wrt_output) / Node(self.abs_val_grad) + bf.func.math.log(self.abs_val_grad)

    def construct_graph(self):
        # TODO: this simply doesn't work so if I want an actual graph visualizer I should fix this
        inps = [self]
        visited = set()
        curr_node_name = self.name
        self.graph.node(curr_node_name)
        while inps:
            curr_node = inps.pop()
            curr_node_name = curr_node.name if isinstance(curr_node, Node) else str(curr_node)

            if isinstance(curr_node, Node):
                for inp in curr_node.inputs:
                    print(inp)
                    inp_name = inp.name if isinstance(inp, Node) else str(inp)
                    inp_label = (
                        inp_name.split(" ")[0][1:] if isinstance(inp, Node) and inp_name.startswith("(") else inp_name
                    )
                    if inp_name not in visited:
                        self.graph.node(inp_name, label=inp_label)
                        self.graph.edge(inp_name, curr_node_name)
                        inps.append(inp)
                        visited.add(inp_name)

        return self.graph

    def visualize(self):
        self.graph.render(view=True)


def reshape_adjoint(adj, shape):
    """
    Convert an adjoint into the appropriate shape before accumulating it into a Node's gradient.

    This is needed whenever a forward operation in the graph performs broadcasting.
    Consider adding a tensor of shape (4, 3, 3) to a tensor of shape (3, 3):

    (4, 3, 3) + (3, 3) --> (4, 3, 3) + (1, 3, 3) --> (4, 3, 3)  [via broadcasting]

    The gradient at the output node will have shape (4, 3, 3).
    Computing a point-wise adjoint for the input gives a gradient of shape (4, 3, 3), but the input
        gradient expects a tensor of size (3, 3).
    Thus, we need to sum-reduce the adjoint along the 0-th axis to produce a (3, 3) adjoint.

    Why is sum-reduction the correct thing to do? Broadcasting could be explicitly represented in
        the computation graph as a 'stack' operation that takes in 4 copies of the (3, 3) input
        tensor and produces the (4, 3, 3) output tensor by stacking them along the 0-th axis.
    The stack operation uses the (3, 3) input 4 times, which would result in the adjoint it produces
        being accumulated into the gradient of that input 4 times, which is equivalent to the sum-
        reduction performed by this function.
    In otherwords: this function is a shortcut for adding a 'stack' operation into the graph
        whenever broadcasting occurs.

    Args:
        adj: the adjoint tensor to be reshaped.
        shape: the new shape that the adjoint tensor should have after 'un-broadcasting'

    Returns:
        adj, summed along dimensions which were produced via broadcasting.
    """
    if adj.shape == shape:
        return adj
    # Should never happen (because broadcasting only adds dimensions; it doesn't remove them)
    if len(shape) > len(adj.shape):
        raise ValueError(f"Cannot reshape adjoint with shape {adj.shape} to accumulate into shape {shape}")
    # Sum along matching dimensions
    for i in range(1, len(shape) + 1):
        if adj.shape[-i] != shape[-i]:
            if shape[-i] == 1:
                adj = np.sum(adj, axis=-i, keepdims=True)
            else:
                raise ValueError(f"Cannot reshape adjoint with shape {adj.shape} to accumulate into shape {shape}")
    # Sum along extra dimensions that adj has
    if len(adj.shape) > len(shape):
        for i in range(len(adj.shape) - len(shape)):
            adj = np.sum(adj, axis=0)
    return adj
