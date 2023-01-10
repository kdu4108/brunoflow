import numpy as np
from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from ..ad import Node
from brunoflow.func.activations import leakyrelu
from brunoflow.func.math import sqrt
from brunoflow.func.reductions import reduce_mean
from brunoflow.func.layers import dropout
from typing import List, Optional, Union


class Dropout(Network):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = bf.net.Dropout(p=0.2)
        >>> input = jnp.random.normal(shape=(20, 16))
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """
    typename = "Dropout"

    def __init__(self, p: float = 0.5, extra_name=None) -> None:
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        # self.training = False
        self.extra_name = extra_name

    def forward(self, x: Node) -> Node:
        return dropout(x, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.p)
