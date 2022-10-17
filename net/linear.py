import math
import numpy as np
from .network import Network, Parameter
from ..func import matmul


def xavier2(m, n):
    return math.sqrt(2 / (m + n))


def xavier1(n):
    return math.sqrt(1 / n)


class Linear(Network):
    def __init__(self, input_size, output_size, name="linear"):
        m = input_size
        n = output_size
        self.W = Parameter(np.random.normal(scale=xavier2(m, n), size=(m, n)), name=f"{name}_w")
        self.b = Parameter(np.random.normal(scale=xavier1(n), size=(n,)), name=f"{name}_b")

    def forward(self, x):
        return matmul(x, self.W) + self.b

    def set_weights(self, W: np.ndarray):
        assert W.shape == self.W.val.shape
        self.W.val = W

    def set_bias(self, b: np.ndarray):
        assert b.shape == self.b.val.shape
        self.b.val = b


class LinearInitToOne(Linear):
    def __init__(self, input_size, output_size):
        super(LinearInitToOne, self).__init__(input_size, output_size)
        self.W = Parameter(np.ones(shape=(input_size, output_size)))
        self.b = Parameter(np.ones(shape=(output_size,)))
