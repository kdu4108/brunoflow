import math
from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from .dropout import Dropout
from ..func import matmul, matrix_transpose, squeeze


def xavier2(m, n):
    return math.sqrt(2 / (m + n))


def xavier1(n):
    return math.sqrt(1 / n)


class Linear(Network):

    typename = "Linear"

    def __init__(self, input_size, output_size, bias=True, random_key_val=42, name="linear", extra_name=None):
        super(Linear, self).__init__()
        random_key = random.PRNGKey(random_key_val)
        # Subkeys are destined for immediate consumption by random functions, while the key is retained to generate more randomness later.
        # See https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html for more info on JAX + random numbers
        random_key2, subkey2 = random.split(random_key)

        m = input_size
        n = output_size
        self.weight = Parameter(random.normal(key=subkey2, shape=(n, m)) * xavier2(n, m), name=f"{name}_w")
        if bias:
            self.bias = Parameter(random.normal(key=random_key, shape=(n,)) * xavier1(n), name=f"{name}_b")
        else:
            self.register_parameter("bias", None)

        self.extra_name = extra_name

    def forward(self, x):
        out = matmul(x, matrix_transpose(self.weight))
        if self.bias is not None:
            out += self.bias
        return out

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


class LinearWithDropout(Network):
    typename = "LinearWithDropout"

    def __init__(self, input_size=32, output_size=1, dropout_prob=0.0):
        super(LinearWithDropout, self).__init__()
        self.dropout = Dropout(p=dropout_prob)
        self.linear = Linear(input_size, output_size=output_size, bias=False, name="ff1")

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear(x)
        return squeeze(out)
