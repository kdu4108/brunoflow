from __future__ import annotations
from colorama import Style
import graphviz
import networkx as nx
import numpy as np
import numpy.typing as npt
import jax
from jax import numpy as jnp
from typing import Any, List, Tuple, Union, Set
import uuid

import brunoflow as bf
from brunoflow.ad.utils import (
    abs_val_grad_product_fn,
    entropy_product_fn,
    max_pos_grad_product_fn,
    max_neg_grad_product_fn,
)


# class IdState:
#     # Stateful way of keeping track of next available id
#     def __init__(self):
#         self.next_available_id = 0


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
            (empty for leaf Nodes). These may be Nodes themselves, jnp.ndarrays, or other types
        backward_func: A function which takes self.grad and computes gradients for each
            of self.inputs (None for leaf Nodes)
        shape (tuple): The same as numpy.ndarray.shape
        size (int): The same as numpy.ndarray.size
        ndim (int): The same as numpy.ndarray.ndim
    """

    def __init__(
        self,
        val: Union[float, jnp.ndarray],
        backward_func=None,
        inputs: List[Union[Node, jnp.ndarray, Any]] = [],
        name: str = None,
        module: bf.net.Network = None,
    ):
        if isinstance(val, np.ndarray):
            # Force all np arrays to be JAX
            val = jnp.array(val)
        self.val = val
        self.grad: Union[jnp.ndarray, jnp.float32] = jnp.zeros_like(val, dtype=jnp.float32)
        self.max_grad_of_output_wrt_node: Tuple[Union[jnp.ndarray, jnp.float32], Union[npt.NDArray[Node], Node]] = (
            jnp.full_like(val, fill_value=-jnp.inf, dtype=jnp.float32),
            np.full(np.shape(val), None),
            # jnp.empty_like(val, dtype=type(None)),
        )
        self.max_neg_grad_of_output_wrt_node: Tuple[Union[jnp.ndarray, jnp.float32], Union[npt.NDArray[Node], Node]] = (
            jnp.full_like(val, fill_value=jnp.inf, dtype=jnp.float32),
            np.full(np.shape(val), None),
            # jnp.empty_like(val, dtype=type(None)),
        )
        self.entropy_wrt_output: Union[jnp.ndarray, jnp.float32] = jnp.zeros_like(val, dtype=jnp.float32)
        self.abs_val_grad: Union[jnp.ndarray, jnp.float32] = jnp.zeros_like(val, dtype=jnp.float32)
        self.backward_func = backward_func
        self.inputs: List[Union[Node, jnp.ndarray]] = inputs
        self.parents: Set[Node] = set()  # parents from the backward pass perspective; the opposite of inputs
        self.module: bf.net.Network = module  # if the Node is a parameter of a Module, save which one here
        self.name: str = name
        self.__num_uses: int = 0
        self.id: str = uuid.uuid4().hex
        self.graph = graphviz.Digraph("computation_graph", filename="computation_graph.gv")

    def copy_(self, new_node: Node):
        if not isinstance(new_node, Node):
            raise ValueError(
                f"ERROR: {self}.copy_ expects new_node to be of type bf.Node, instead received object of type {type(value)}"
            )

        value = new_node.val
        if not isinstance(value, jnp.ndarray):
            print(
                f"WARNING: {self}.copy_ expects value of new_nodes to be of type jnp.ndarray, instead received object of type {type(value)}"
            )

        self.val = value

    def get_id(self):
        return self.id

    def id(self, next_available_id: int = 0):
        if self.id is None:
            self.id = next_available_id
            next_available_id += 1

        return self.id, next_available_id

    def subtree(self, visited=set()) -> Tuple[List[Node], Tuple[Node, Node]]:
        """
        Given a root node, return a list of all nodes in its subtree and all edges that make up the subtree.
        """
        nodes: List[Node] = [self]
        edges: List[Tuple[Node, Node]] = []
        if self in visited:
            return [], []

        if len(self.inputs) > 0:
            for input_node in self.inputs:
                if isinstance(input_node, Node):
                    edges.append((input_node, self))
                    # recurse
                    local_nodes, local_edges = input_node.subtree(visited=visited)
                    nodes.extend(local_nodes)
                    edges.extend(local_edges)
                    visited.add(input_node)
                # Commenting out this else bit for now because it's annoying to have nparrays show up because of reshape(axis=blah)
                # else:
                #     # create node for numpy array
                #     edges.append((Node(input_node, name="nparray"), self))
                #     nodes.append(Node(input_node, name="nparray"))

        return nodes, edges

    def visualize(
        self, save_path="graph.png", vals_to_include={"grad", "entropy"}, shape_only=True, collapse_to_modules=False
    ):
        import matplotlib.pyplot as plt

        nodes, edges = self.subtree(visited=set())
        G = nx.DiGraph(directed=True)

        def get_node_shortname(node: Union[Node, bf.net.Network]) -> str:
            if isinstance(node, bf.net.Network):
                return node._get_name()
            elif isinstance(node, Node):
                name = node.name
                if isinstance(name, str):
                    return name.split(" ", maxsplit=1)[0][1:] if name.startswith("(") else name
                elif isinstance(node.val, jnp.ndarray):
                    name = str(node.shape)
                else:
                    raise NotImplementedError(
                        f"should this ever happen? get_node_shortname was passed a node with value of type {type(node.val)} instead of jnparray"
                    )
            else:
                raise NotImplementedError(
                    f"should this ever happen? get_node_shortname was passed obj of type {type(node)} instead of a Node"
                )

            return name

        def get_grad(node: Node) -> jnp.ndarray:
            grad = node.grad.copy()
            if isinstance(grad, jnp.ndarray) and len(grad.shape) > 1:
                grad = jnp.mean(grad, axis=0)
            return grad.round(decimals=4)

        def get_entropy(node: Node) -> jnp.ndarray:
            entropy = node.compute_entropy().val.copy()
            # if isinstance(entropy, np.ndarray) and "_w" not in node.name:
            #     entropy = np.mean(entropy, axis=0)
            return entropy.round(decimals=4)

        def get_label_from_pygraph_node(pygraph_node):
            import re

            node_name = re.split(" |,", pygraph_node.name)[1]
            operator = node_name[1:] if node_name.startswith("(") else node_name
            return operator

        def construct_label(
            node: Union[Node, bf.net.Network], vals_to_include={"grad", "entropy"}, shape_only=True
        ) -> str:
            node_name: str = get_node_shortname(node)
            label = node_name
            if isinstance(node, bf.net.Network):
                return label

            if shape_only:
                label += f"\nshape: {node.shape}"
            else:
                if "grad" in vals_to_include:
                    node_grad = get_grad(node)
                    label += f"\ngrad: {node_grad}"
                if "entropy" in vals_to_include:
                    node_entropy = get_entropy(node)
                    label += f"\nentropy: {node_entropy}"

            return label

        def construct_edge_label(
            e1: Union[Node, bf.net.Network],
            e2: Union[Node, bf.net.Network],
            vals_to_include={"grad", "entropy"},
            shape_only=True,
        ) -> str:
            e1_name: str = get_node_shortname(e1)
            e2_name: str = get_node_shortname(e2)
            label = f"{e1_name} -> {e2_name}"

            return label

        def collapse_nodes_to_modules(nodes, edges):
            node_to_modules_map = dict()
            new_nodes = []
            for node in nodes:
                if node.module is not None:
                    new_nodes.append(node.module)
                    node_to_modules_map[node] = node.module
                else:
                    new_nodes.append(node)

            new_edges = []
            for src, dest in edges:
                new_src, new_dest = src, dest
                if src in node_to_modules_map:
                    new_src = node_to_modules_map[src]
                if dest in node_to_modules_map:
                    new_dest = node_to_modules_map[dest]

                if hasattr(new_src, "id") and hasattr(new_dest, "id"):
                    if new_src.id != new_dest.id:
                        new_edges.append((new_src, new_dest))
                elif new_src != new_dest:
                    new_edges.append((new_src, new_dest))

            return new_nodes, new_edges

        def convert_nodes_and_edges_to_json(nodes, edges):
            new_nodes = [
                {"id": node.id, "name": node.name, "val": float(node.max_grad_of_output_wrt_node[0].mean().__array__())}
                for node in nodes
            ]
            new_edges = [{edge[0].id: edge[1].id} for edge in edges]
            import json

            with open("nodes.json", "w") as f:
                json.dump(new_nodes, f)
            with open("edges.json", "w") as f:
                json.dump(new_edges, f)

        if collapse_to_modules:
            nodes, edges = collapse_nodes_to_modules(nodes, edges)

        convert_nodes_and_edges_to_json(nodes, edges)
        # G.add_nodes_from(nodes)
        G.add_nodes_from(
            [
                (node, {"label": construct_label(node, vals_to_include=vals_to_include, shape_only=shape_only)})
                for node in nodes
            ]
        )
        # G.add_edges_from([(e1, e2, {"label": construct_edge_label(e1, e2)}) for (e1, e2) in edges])  # [(n1.get_id(), n2.get_id()) for n1, n2 in edges])
        G.add_edges_from(edges)

        G = nx.nx_agraph.to_agraph(G)
        # for i in range(len(G.nodes())):
        #     G.nodes()[i].attr["label"] = get_label_from_pygraph_node(G.nodes()[i])
        G.layout(prog="dot")
        G.draw(path=save_path)

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

    def get_parents(self) -> List[Node]:
        return list(self.parents)

    def get_max_grad_parent(self) -> Union[npt.NDArray[Node], Node]:
        return self.max_grad_of_output_wrt_node[1]

    def get_max_neg_grad_parent(self) -> Union[npt.NDArray[Node], Node]:
        return self.max_neg_grad_of_output_wrt_node[1]

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

    def backprop(
        self,
        # values_to_compute=("grad", "abs_val_grad", "entropy"),
        values_to_compute=("grad", "max_grad", "abs_val_grad", "entropy"),
        save_leaf_grads_only=False,
        verbose=False,
    ):
        """
        Initiate a backward pass starting from this Node.
        Computes gradients from every Node reachable from this Node, back to the leaves.
        """
        # print(f"Entering backprop on node {self.name}:", self)
        if isinstance(self.grad, jnp.ndarray):
            self.grad = jnp.full_like(self.grad, 1.0)
            self.max_grad_of_output_wrt_node = (
                jnp.full_like(self.max_grad_of_output_wrt_node[0], 1.0),
                np.full(self.max_grad_of_output_wrt_node[1].shape, None),
            )
            self.max_neg_grad_of_output_wrt_node = (
                jnp.full_like(self.max_neg_grad_of_output_wrt_node[0], 1.0),
                np.full(self.max_neg_grad_of_output_wrt_node[1].shape, None),
            )
            self.entropy_wrt_output = jnp.full_like(self.entropy_wrt_output, 0.0)
            self.abs_val_grad = jnp.full_like(self.abs_val_grad, 1.0)
        else:
            # If the grad is a float rather than an jnp.ndarray
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
        self.__backprop(values_to_compute=values_to_compute, verbose=verbose, save_leaf_grads_only=save_leaf_grads_only)

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
                prev_grad = jnp.copy(self.inputs[i].grad)
                self.inputs[i].grad += adjoint_gradient
                if verbose:
                    print(
                        f"    {Style.RESET_ALL}Updating grad of node {self.inputs[i].name}:\n     from {prev_grad} \n       to {self.inputs[i].grad}."
                    )
                del prev_grad

    def _compute_and_accumulate_max_pos_and_neg_grads_for_inputs(self, input_vals, backward_func, verbose=False):
        assert (self.max_grad_of_output_wrt_node[0] >= self.max_neg_grad_of_output_wrt_node[0]).all()

        # Multiply the max pos grad of self w.r.t. output by the local derivative for each input val
        adjoints_max_grad = backward_func(
            self.val,
            dict(out_max_pos_grad=self.max_grad_of_output_wrt_node[0]),
            *input_vals,
            product_fn=max_pos_grad_product_fn,
        )

        # Multiply the max neg grad of self w.r.t. output by the local derivative for each input val
        adjoints_max_neg_grad = backward_func(
            self.val,
            dict(out_max_neg_grad=self.max_neg_grad_of_output_wrt_node[0]),
            *input_vals,
            product_fn=max_neg_grad_product_fn,
        )

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
                self.inputs[i].parents.add(self)
                # An adjoint may need to be 'reshaped' before it can be accumulated if the forward operation used broadcasting.
                adj_max_grad = reshape_adjoint(adj_max_grad, self.inputs[i].grad.shape)
                adj_max_neg_grad = reshape_adjoint(adj_max_neg_grad, self.inputs[i].grad.shape)

                # If the max pos grad is less than the max neg grad, that means there was some sign flipping, and we should reverse which is most positive and most negative. Use jnp.maximum and jnp.minimum to do this element-wise.
                adj_max_grad, adj_max_neg_grad = jnp.maximum(adj_max_grad, adj_max_neg_grad), jnp.minimum(
                    adj_max_grad, adj_max_neg_grad
                )

                # ACCUMULATE MAX POS GRAD BY USING THE MAX OPERATOR
                # Replace the current (max grad value, parent) of the ith input node with (max grad value of self, self) for all elements in the ith input node that have higher max grad values in self.
                # prev_vals = jnp.copy(self.inputs[i].max_grad_of_output_wrt_node[0])
                prev_vals, prev_parents = jnp.copy(self.inputs[i].max_grad_of_output_wrt_node[0]), np.copy(
                    self.inputs[i].max_grad_of_output_wrt_node[1]
                )

                inds_to_replace_max_grad = adj_max_grad > self.inputs[i].max_grad_of_output_wrt_node[0]
                self.inputs[i].max_grad_of_output_wrt_node = (
                    jnp.where(
                        inds_to_replace_max_grad,
                        adj_max_grad,
                        self.inputs[i].max_grad_of_output_wrt_node[0],
                    ),
                    np.where(
                        inds_to_replace_max_grad,
                        self,
                        self.inputs[i].max_grad_of_output_wrt_node[1],
                    ),
                )

                if verbose and not (
                    jnp.array_equal(prev_vals, self.inputs[i].max_grad_of_output_wrt_node[0])
                    and np.array_equal(prev_parents, self.inputs[i].max_grad_of_output_wrt_node[1])
                ):
                    print(
                        f"    Replacing max_grad of node {self.inputs[i].name}:\n     from {(prev_vals, prev_parents)} \n       to {self.inputs[i].max_grad_of_output_wrt_node}."
                    )

                del prev_vals, prev_parents

                # ACCUMULATE MAX NEG GRAD BY USING THE MIN OPERATOR
                # Replace the current (max neg grad value, parent) of the ith input node with (max neg grad value of self, self) for all elements in the ith input node that have more negative max neg grad values in self.
                # prev_vals = jnp.copy(self.inputs[i].max_neg_grad_of_output_wrt_node[0])
                prev_vals, prev_parents = jnp.copy(self.inputs[i].max_neg_grad_of_output_wrt_node[0]), np.copy(
                    self.inputs[i].max_neg_grad_of_output_wrt_node[1]
                )

                inds_to_replace_max_neg_grad = adj_max_neg_grad < self.inputs[i].max_neg_grad_of_output_wrt_node[0]
                self.inputs[i].max_neg_grad_of_output_wrt_node = (
                    jnp.where(
                        inds_to_replace_max_neg_grad,
                        adj_max_neg_grad,
                        self.inputs[i].max_neg_grad_of_output_wrt_node[0],
                    ),
                    np.where(
                        inds_to_replace_max_neg_grad,
                        self,
                        self.inputs[i].max_neg_grad_of_output_wrt_node[1],
                    ),
                )

                if verbose and not (
                    jnp.array_equal(prev_vals, self.inputs[i].max_neg_grad_of_output_wrt_node[0])
                    and np.array_equal(prev_parents, self.inputs[i].max_neg_grad_of_output_wrt_node[1])
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
            product_fn=abs_val_grad_product_fn,
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
                prev_abs_val_grad = jnp.copy(self.inputs[i].abs_val_grad)
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
            product_fn=entropy_product_fn,
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
                prev_entropy = jnp.copy(self.inputs[i].entropy_wrt_output)
                self.inputs[i].entropy_wrt_output += adj_entropy
                if verbose:
                    print(
                        f"    {Style.RESET_ALL}Updating unnormalized entropy of node {self.inputs[i].name}:\n     from {prev_entropy} \n       to {self.inputs[i].entropy_wrt_output}."
                    )
                    print(
                        f"    {Style.RESET_ALL}Updating actual entropy of node {self.inputs[i].name}:\n     to {self.inputs[i].compute_entropy()}."
                    )
                del prev_entropy

    def __backprop(
        self,
        values_to_compute=("grad", "max_grad", "abs_val_grad", "entropy"),
        save_leaf_grads_only=False,
        verbose=False,
    ):
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
                if save_leaf_grads_only and self.inputs:
                    self.none_out_self_gradients()
                    # self.zero_self_gradients()

            # Continue recursively backpropagating
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__backprop(
                        values_to_compute=values_to_compute, verbose=verbose, save_leaf_grads_only=save_leaf_grads_only
                    )

    def zero_self_gradients(self):
        if isinstance(self.grad, jnp.ndarray):
            self.grad = jnp.full_like(self.grad, 0.0)

            # Careful - ndarray.fill() when type(ndarray) is int results in this int overflow thing instead of inf.
            self.max_grad_of_output_wrt_node = (
                jnp.full_like(self.max_grad_of_output_wrt_node[0], fill_value=-jnp.inf, dtype=jnp.float32),
                np.full(self.max_grad_of_output_wrt_node[1].shape, None),
                # jnp.empty_like(self.max_grad_of_output_wrt_node[1], dtype=type(None)),
            )
            self.max_neg_grad_of_output_wrt_node = (
                jnp.full_like(self.max_neg_grad_of_output_wrt_node[0], fill_value=jnp.inf, dtype=jnp.float32),
                np.full(self.max_neg_grad_of_output_wrt_node[1].shape, None),
                # jnp.empty_like(self.max_neg_grad_of_output_wrt_node[1], dtype=type(None)),
            )
            self.entropy_wrt_output = jnp.full_like(self.entropy_wrt_output, 0.0)
            self.abs_val_grad = jnp.full_like(self.abs_val_grad, 0.0)
        else:
            self.grad = 0.0
            self.max_grad_of_output_wrt_node = (0.0, None)
            self.max_neg_grad_of_output_wrt_node = (0.0, None)
            self.entropy_wrt_output = 0.0
            self.abs_val_grad = 0.0

    def none_out_self_gradients(self):
        self.grad = None
        self.max_grad_of_output_wrt_node = (None, None)
        self.max_neg_grad_of_output_wrt_node = (None, None)
        self.entropy_wrt_output = None
        self.abs_val_grad = None

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
            if isinstance(self.grad, jnp.ndarray):
                self.grad = jnp.full_like(self.grad, 0.0)
                # Careful - ndarray.fill() when type(ndarray) is int results in this int overflow thing instead of inf.
                self.max_grad_of_output_wrt_node = (
                    jnp.full_like(self.max_grad_of_output_wrt_node[0], fill_value=-jnp.inf, dtype=jnp.float32),
                    np.full(self.max_grad_of_output_wrt_node[1].shape, None),
                    # jnp.empty_like(self.max_grad_of_output_wrt_node[1], dtype=type(None)),
                )
                self.max_neg_grad_of_output_wrt_node = (
                    jnp.full_like(self.max_neg_grad_of_output_wrt_node[0], fill_value=jnp.inf, dtype=jnp.float32),
                    np.full(self.max_neg_grad_of_output_wrt_node[1].shape, None),
                    # jnp.empty_like(self.max_neg_grad_of_output_wrt_node[1], dtype=type(None)),
                )
                self.entropy_wrt_output = jnp.full_like(self.entropy_wrt_output, 0.0)
                self.abs_val_grad = jnp.full_like(self.abs_val_grad, 0.0)

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
                adj = jnp.sum(adj, axis=-i, keepdims=True)
            else:
                raise ValueError(f"Cannot reshape adjoint with shape {adj.shape} to accumulate into shape {shape}")
    # Sum along extra dimensions that adj has
    if len(adj.shape) > len(shape):
        for i in range(len(adj.shape) - len(shape)):
            adj = jnp.sum(adj, axis=0)
    return adj
