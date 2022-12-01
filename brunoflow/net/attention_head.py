from .linear import Linear
from .network import Network
from brunoflow.func import reduce_mean, matmul, leakyrelu, concat, stack


class AttentionHead(Network):
    def __init__(self):
        # Embedding layers
        self.q = Linear(768, 64, name="q")
        self.k = Linear(768, 64, name="k")
        self.v = Linear(768, 64, name="v")

    def forward(self, x):
        # x has dim (bs, 768)
        # output has dim (bs, 64)
        return reduce_mean(stack([matmul(x, self.q), matmul(x, self.k), matmul(x, self.v)], axis=0), axis=0)


class AttentionLayer(Network):
    def __init__(self):
        # Embedding layers
        self.att_head1 = AttentionHead()
        self.att_head2 = AttentionHead()
        self.att_head3 = AttentionHead()
        self.att_head4 = AttentionHead()
        self.att_head5 = AttentionHead()
        self.att_head6 = AttentionHead()
        self.att_head7 = AttentionHead()
        self.att_head8 = AttentionHead()
        self.att_head9 = AttentionHead()
        self.att_head10 = AttentionHead()
        self.att_head11 = AttentionHead()
        self.att_head12 = AttentionHead()
        self.ff = Linear(768, 768)
        self.pos_ff = Linear(768, 3072)
        self.pos_ff2 = Linear(3072, 768)

    def forward(self, x):
        # x has dim (bs, 768)
        bs_by_768 = concat(
            [
                self.att_head1(x),
                self.att_head2(x),
                self.att_head3(x),
                self.att_head4(x),
                self.att_head5(x),
                self.att_head6(x),
                self.att_head7(x),
                self.att_head8(x),
                self.att_head9(x),
                self.att_head10(x),
                self.att_head11(x),
                self.att_head12(x),
            ],
            axis=1,
        )
        bs_by_768 = matmul(bs_by_768, self.ff)
        bs_by_768 = matmul(bs_by_768, self.pos_ff)
        bs_by_768 = matmul(bs_by_768, self.pos_ff2)

        return bs_by_768
