from brunoflow.net import Network
from brunoflow.ad import Node
from brunoflow.opt import cross_entropy_loss


class CrossEntropyLoss(Network):
    r"""This criterion computes the cross entropy loss between input and target."""

    def forward(self, output: Node, target: Node) -> Node:
        return cross_entropy_loss(output, target)
