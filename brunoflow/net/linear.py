import math
from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from ..func import matmul


def xavier2(m, n):
    return math.sqrt(2 / (m + n))


def xavier1(n):
    return math.sqrt(1 / n)


class Linear(Network):
    def __init__(self, input_size, output_size, random_key_val=42, name="linear"):
        random_key = random.PRNGKey(random_key_val)
        # Subkeys are destined for immediate consumption by random functions, while the key is retained to generate more randomness later.
        # See https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html for more info on JAX + random numbers
        random_key2, subkey2 = random.split(random_key)

        m = input_size
        n = output_size
        self.b = Parameter(random.normal(key=random_key, shape=(n,)) * xavier1(n), name=f"{name}_b")
        self.W = Parameter(random.normal(key=subkey2, shape=(m, n)) * xavier2(m, n), name=f"{name}_w")

    def forward(self, x):
        return matmul(x, self.W) + self.b

    def set_weights(self, W: jnp.ndarray):
        assert W.shape == self.W.val.shape
        self.W.val = W

    def set_bias(self, b: jnp.ndarray):
        assert b.shape == self.b.val.shape
        self.b.val = b


class LinearInitToOne(Linear):
    def __init__(self, input_size, output_size):
        super(LinearInitToOne, self).__init__(input_size, output_size)
        self.W = Parameter(jnp.ones(shape=(input_size, output_size)))
        self.b = Parameter(jnp.ones(shape=(output_size,)))
