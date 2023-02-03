from .linear import Linear
from .dropout import Dropout
from .network import Network
from brunoflow.func.activations import leakyrelu


class MLP(Network):
    typename = "MLP"

    def __init__(self, input_size=784, hidden_size=500, num_classes=10, dropout_prob=0.0):
        super(MLP, self).__init__()
        self.ff1 = Linear(input_size, hidden_size, name="ff1")
        self.dropout = Dropout(p=dropout_prob)
        self.ff2 = Linear(hidden_size, num_classes, name="ff2")

    def forward(self, x):
        out = self.ff1(x)
        out = leakyrelu(out)
        out = self.dropout(out)
        out = self.ff2(out)
        return out
