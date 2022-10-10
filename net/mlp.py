from .linear import Linear
from .network import Network
from brunoflow.func.activations import relu


class MLP(Network):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        self.ff1 = Linear(input_size, hidden_size)
        self.ff2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.ff1(x)
        out = relu(out)
        out = self.ff2(out)
        return out
