from collections import namedtuple
from jax import numpy as jnp
import torch
from typing import Union
from ..ad import Node


class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def make_into_bf_node(t: Union[Node, torch.Tensor]):
    if isinstance(t, torch.Tensor):
        return Node(val=jnp.array(t.numpy()))
    elif isinstance(t, Node):
        return t
    else:
        raise ValueError(
            f"make_into_bf_node expects an input of type `torch.Tensor` or `bf.Node`, instead received {type(t)}."
        )
