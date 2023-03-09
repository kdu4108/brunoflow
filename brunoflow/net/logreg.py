import math
from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from .dropout import Dropout
from .linear import Linear
from ..func import matmul, matrix_transpose, squeeze, sigmoid


def xavier2(m, n):
    return math.sqrt(2 / (m + n))


def xavier1(n):
    return math.sqrt(1 / n)


class LogReg(Network):
    typename = "LogReg"

    def __init__(self, input_size=30, output_size=2, dropout_prob=0.0, bias=True):
        super(LogReg, self).__init__()
        self.dropout = Dropout(p=dropout_prob)
        self.linear = Linear(input_size, output_size=output_size, bias=bias, name="ff1")

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear(out)
        return sigmoid(out)
