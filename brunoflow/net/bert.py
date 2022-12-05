from .linear import Linear
from .network import Network
from brunoflow.net.attention_head import AttentionLayer
from brunoflow.func import leakyrelu, matmul


class BERT(Network):
    def __init__(self, num_layers=12, vocab_size=30522):
        # Embedding layers
        self.w_emb = Linear(vocab_size, 768, name="w_emb")
        self.p_emb = Linear(768, 512, name="p_emb")

        # Attn layer
        self.att_layers = [AttentionLayer() for _ in range(num_layers)]

    def forward(self, x):
        # x = (seq_len, 30522)
        out = self.w_emb(x)
        for att_layer in self.att_layers:
            out = att_layer(out)

        return out
