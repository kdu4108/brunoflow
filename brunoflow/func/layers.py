"""
This module defines functions that operate on inputs to a layer in a network.
"""
import numpy as np
from ..ad import Node


def dropout(x: Node, p: float = 0.5, training: bool = True) -> Node:
    if not training:
        return x

    mask = np.random.choice([0, 1], size=x.shape, p=[p, 1 - p])

    # Use inverted dropout (multiply by scaling factor) to account for the total activations being lower in training than testing
    # when dropping out self.p neurons.
    # https://stats.stackexchange.com/questions/205932/dropout-scaling-the-activation-versus-inverting-the-dropout

    scaling_factor = 1 / (1 - p)
    return (x * mask) * scaling_factor
