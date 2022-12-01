from .linear import Linear
from .network import Network
from brunoflow.net.attention_head import AttentionLayer
from brunoflow.func import leakyrelu, matmul


class BERT(Network):
    def __init__(self):
        # Embedding layers
        self.w_emb = Linear(30522, 768, name="w_emb")
        self.p_emb = Linear(768, 512, name="p_emb")

        # Attn layer
        self.att_layer1 = AttentionLayer()
        self.att_layer2 = AttentionLayer()
        self.att_layer3 = AttentionLayer()
        self.att_layer4 = AttentionLayer()
        self.att_layer5 = AttentionLayer()
        self.att_layer6 = AttentionLayer()
        self.att_layer7 = AttentionLayer()
        self.att_layer8 = AttentionLayer()
        self.att_layer9 = AttentionLayer()
        self.att_layer10 = AttentionLayer()
        self.att_layer11 = AttentionLayer()
        self.att_layer12 = AttentionLayer()

    def forward(self, x):
        # x = (seq_len, 30522)
        emb = matmul(x, self.w_emb)
        out = self.att_layer1(emb)
        out = self.att_layer2(out)
        out = self.att_layer3(out)
        out = self.att_layer4(out)
        out = self.att_layer5(out)
        out = self.att_layer6(out)
        out = self.att_layer7(out)
        out = self.att_layer8(out)
        out = self.att_layer9(out)
        out = self.att_layer10(out)
        out = self.att_layer11(out)
        out = self.att_layer12(out)

        return out
