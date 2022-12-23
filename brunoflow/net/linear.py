import math
from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from ..func import matmul, matrix_transpose


def xavier2(m, n):
    return math.sqrt(2 / (m + n))


def xavier1(n):
    return math.sqrt(1 / n)


class Linear(Network):

    typename = "Linear"

    def __init__(self, input_size, output_size, random_key_val=42, name="linear"):
        super(Linear, self).__init__()
        random_key = random.PRNGKey(random_key_val)
        # Subkeys are destined for immediate consumption by random functions, while the key is retained to generate more randomness later.
        # See https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html for more info on JAX + random numbers
        random_key2, subkey2 = random.split(random_key)

        m = input_size
        n = output_size
        self.bias = Parameter(random.normal(key=random_key, shape=(n,)) * xavier1(n), name=f"{name}_b")
        self.weight = Parameter(random.normal(key=subkey2, shape=(n, m)) * xavier2(n, m), name=f"{name}_w")

    def forward(self, x):
        return matmul(x, matrix_transpose(self.weight)) + self.bias

    def set_weights(self, W: jnp.ndarray):
        assert W.shape == self.weight.val.shape
        self.weight.val = W

    def set_bias(self, b: jnp.ndarray):
        assert b.shape == self.bias.val.shape
        self.bias.val = b


class LinearInitToOne(Linear):
    def __init__(self, input_size, output_size):
        super(LinearInitToOne, self).__init__(input_size, output_size)
        self.weight = Parameter(jnp.ones(shape=(input_size, output_size)))
        self.bias = Parameter(jnp.ones(shape=(output_size,)))
