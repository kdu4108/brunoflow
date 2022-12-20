from jax import numpy as jnp
from jax import random
from .network import Network, Parameter
from ..ad import Node
from brunoflow.func.activations import leakyrelu
from typing import Optional


class Embedding(Network):
    """
    Implementation of an Embedding Layer. It's missing a lot of functionality (including no special zeroing-out gradients of padding_idx)
    TODO(kdu): somehow ensure gradient of pad idx is 0.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[jnp.ndarray] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
        random_key_val=42,
    ) -> None:
        random_key = random.PRNGKey(random_key_val)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.name = f"Embedding Layer ({self.num_embeddings}, {self.embedding_dim})"
        embedding_weights_name = f"emb weights ({self.num_embeddings}, {self.embedding_dim})"
        if _weight is None:
            self.embeddings = Parameter(
                random.normal(key=random_key, shape=(num_embeddings, embedding_dim)), name=embedding_weights_name
            )
            self._fill_padding_idx_with_zero()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.embeddings = Parameter(_weight, name=embedding_weights_name)

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            self.embeddings.val = self.embeddings.val.at[self.padding_idx].set(0)

    def forward(self, indices: jnp.ndarray):
        return self.embeddings[indices]

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.
        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.
        Examples::
            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        return embedding
