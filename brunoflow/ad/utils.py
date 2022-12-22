import jax
from jax import numpy as jnp
from torch import Tensor


@jax.jit
def abs_val_grad_product_fn(l_adj, out_abs_val_grad_dict):
    return jnp.abs(l_adj) * out_abs_val_grad_dict["out_abs_val_grad"]


@jax.jit
def entropy_product_fn(l_adj, out_grad_and_entropy_dict):
    return (jnp.abs(l_adj) * out_grad_and_entropy_dict["out_entropy"]) + jnp.nan_to_num(
        -jnp.abs(l_adj) * jnp.log(jnp.abs(l_adj)) * out_grad_and_entropy_dict["out_abs_val_grad"],
        copy=False,
        nan=0.0,
    )


def check_node_equals_tensor(node, tensor: Tensor):
    return jnp.array_equal(node.val, tensor.detach().numpy())


def check_node_allclose_tensor(node, tensor: Tensor):
    return jnp.allclose(node.val, tensor.detach().numpy())
