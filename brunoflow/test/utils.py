from collections.abc import Iterable
from typing import List
import brunoflow as bf
from jax import numpy as jnp
import numpy as np
import random
import torch
from functools import reduce

from brunoflow.ad.node import Node

rtol = 1e-05
atol = 1e-08


class TestInput:
    def __init__(self, name, vals):
        self.name = name
        self.vals = vals


def inputs_to_params(inp, lift_every_other=False):
    ret = []
    for i, val in enumerate(inp.vals):
        if isinstance(val, int) or (isinstance(val, jnp.ndarray) and val.dtype == int):
            ret.append(val)
        elif not lift_every_other or i % 2 == 0:
            ret.append(bf.Parameter(val))
        else:
            ret.append(val)
    return ret


def inputs_to_torch(inp, requires_grad=False):
    ret = []
    for val in inp.vals:
        if isinstance(val, int):
            ret.append(val)
        elif isinstance(val, float):
            ret.append(torch.tensor([val], requires_grad=requires_grad))
        elif isinstance(val, jnp.ndarray) and val.dtype == int:
            ret.append(torch.tensor(np.array(val)))
        else:
            if isinstance(val, jnp.ndarray):
                ret.append(torch.tensor(np.array(val), requires_grad=requires_grad))
            else:
                if isinstance(val, np.ndarray):
                    print(
                        f"WARNING: input {val} to torch has type np.ndarray instead of jnp.ndarray, so weird things might happen during testing."
                    )
                ret.append(torch.tensor(val, requires_grad=requires_grad))
    return ret


def check(self, x, target):
    if isinstance(x, bf.ad.Node):
        x = x.val
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        if isinstance(x, np.ndarray):
            print(f"WARNING: input x={x} has type nparray instead of jnparray.")
        if x.dtype == bool or x.dtype == int:
            if not jnp.array_equal(x, target.numpy()):
                self.assertIs(x, target.numpy())
        elif x.dtype == float:
            if not jnp.allclose(x, target.numpy(), rtol=rtol, atol=atol):
                self.assertIs(x, target.numpy())
    elif isinstance(x, bool):
        self.assertEqual(x, target.item())
    elif isinstance(x, float):
        if not jnp.allclose(x, target.item(), rtol=rtol, atol=atol):
            self.assertIs(x, target.item())
    elif isinstance(x, Iterable):
        self.assertEqual(len(x), len(target))
        for i in range(len(x)):
            check(self, x[i], target[i])
    else:
        raise TypeError(f"Unknown type of x in check: {type(x)}")


def add_forward_tests(clss, basename, fn, torch_fn, inputs, lift_every_other=False):
    for inp in inputs:

        def test_forward(self):
            # print('======= FORWARD TEST =======')
            x = inputs_to_params(inp, lift_every_other)
            x_t = inputs_to_torch(inp)
            # print('x:', x)
            # print('x_t:', x_t)
            y = fn(*x)
            y_t = torch_fn(*x_t)
            # print('y:', y)
            # print('y_t:', y_t)
            check(self, y, y_t)

        setattr(clss, f"test_forward_{basename}_{inp.name}", test_forward)


def add_backward_tests(clss, basename, fn, torch_fn, inputs, lift_every_other=False):
    for inp in inputs:

        def test_backward(self):
            x = inputs_to_params(inp, lift_every_other)
            x_t = inputs_to_torch(inp, requires_grad=True)
            y = fn(*x)
            y_t = torch_fn(*x_t)
            if not isinstance(y, Iterable):
                y = [y]
                y_t = [y_t]
            for i in range(len(y)):
                y[i].backprop()
                y_t[i].backward(torch.ones_like(y_t[i]), retain_graph=True)
                grads_x = [x_.grad for x_ in x if isinstance(x_, bf.Parameter)]
                grads_xt = [x_t[i].grad for i in range(len(x_t)) if isinstance(x[i], bf.Parameter)]
                check(self, grads_x, grads_xt)
                y[i].zero_gradients()
                # I *think* this is how you're supposed to do it...
                for x_ in x_t:
                    if x_.requires_grad:
                        x_.grad.zero_()

        setattr(clss, f"test_backward_{basename}_{inp.name}", test_backward)


def add_tests(clss, basename, fn, torch_fn, inputs, test_backward=True):

    add_forward_tests(clss, basename, fn, torch_fn, inputs)
    if test_backward:
        add_backward_tests(clss, basename, fn, torch_fn, inputs)
    # Version where some inputs aren't lifted to be Parameters
    if len(inputs[0].vals) > 1:
        add_forward_tests(clss, basename + "_halflifted", fn, torch_fn, inputs, lift_every_other=True)
        if test_backward:
            add_backward_tests(clss, basename + "_halflifted", fn, torch_fn, inputs, lift_every_other=True)


def inputs(input_vals):
    return [TestInput(f"input{i}", vals) for i, vals in enumerate(input_vals)]


def random_inputs(input_shapes, rand=np.random.normal):
    return [
        TestInput(
            reduce(lambda a, b: a + "x" + b, [str(shape) for shape in shapes]),
            [jnp.array(rand(size=shape)) for shape in shapes],
        )
        for shapes in input_shapes
    ]


def get_all_paths_to_node(curr_node: Node) -> List[List[Node]]:
    # Helper for bruteforce computing backprop values
    if not curr_node.inputs:
        return [[curr_node]]
    all_paths_to_curr_node = []
    for inp in curr_node.inputs:
        if isinstance(inp, Node):
            paths_to_inp = get_all_paths_to_node(inp)
            all_paths_to_curr_node_through_inp = [path + [curr_node] for path in paths_to_inp]
            all_paths_to_curr_node += all_paths_to_curr_node_through_inp
    return all_paths_to_curr_node


def get_all_paths_from_src_to_target_node(target_node: Node, src_node: Node) -> List[List[Node]]:
    # Helper for bruteforce computing backprop values
    all_paths = get_all_paths_to_node(target_node)
    all_paths_from_src_to_target = [
        path for path in all_paths if jnp.all((path[0] == src_node).val)
    ]  # Note - equality is simply checking if the value of the two nodes is the same, so be careful!
    print(all_paths_from_src_to_target)
    return all_paths_from_src_to_target


def compute_path_values(paths: List[List[Node]]) -> List[float]:
    pass
