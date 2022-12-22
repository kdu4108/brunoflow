import numbers
from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from ..ad import Node
from brunoflow.func.activations import leakyrelu
from brunoflow.func.math import sqrt
from brunoflow.func.reductions import reduce_mean
from typing import List, Optional, Union


class LayerNorm(Network):
    typename = "LayerNorm"

    def __init__(
        self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(jnp.ones(shape=self.normalized_shape), name="layernorm_W")
            self.bias = Parameter(jnp.zeros(shape=self.normalized_shape), name="layernorm_b")
        # else:
        #     self.register_parameter('weight', None)
        #     self.register_parameter('bias', None)

    def forward(self, x: Node) -> Node:
        # taken from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/layer_norm/__init__.py
        assert self.normalized_shape == x.shape[-len(self.normalized_shape) :]

        # The dimensions to calculate the mean and variance on
        axes = [-(i + 1) for i in range(len(self.normalized_shape))]

        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = reduce_mean(x, axis=tuple(axes), keepdims=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_x2 = reduce_mean(x**2, axis=tuple(axes), keepdims=True)

        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_x2 - mean**2

        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / sqrt(var + self.eps)

        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm
