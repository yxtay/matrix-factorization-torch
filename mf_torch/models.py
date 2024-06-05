from __future__ import annotations

import torch
import torch.nn.functional as F


class MatrixFactorization(torch.nn.Module):
    def __init__(
        self: MatrixFactorization,
        num_embeddings: int,
        embedding_dim: int,
        *,
        _weight: torch.Tensor | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        mode: str = "sum",
        sparse: bool = False,
        padding_idx: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            _weight=_weight,
            max_norm=max_norm,
            norm_type=norm_type,
            mode=mode,
            sparse=sparse,
            padding_idx=padding_idx,
        )
        self.normalize = normalize

    @property
    def sparse(self: MatrixFactorization) -> bool:
        return self.embedding.sparse

    @classmethod
    def from_pretrained(
        cls: MatrixFactorization,
        embeddings: torch.Tensor,
        *,
        freeze: bool = True,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        mode: str = "sum",
        sparse: bool = False,
        padding_idx: int = 0,
        normalize: bool = True,
    ) -> MatrixFactorization:
        assert embeddings.dim() == 2
        rows, cols = embeddings.shape
        model = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            max_norm=max_norm,
            norm_type=norm_type,
            mode=mode,
            sparse=sparse,
            padding_idx=padding_idx,
            normalize=normalize,
        )
        model.embedding_bag.weight.requires_grad = not freeze
        return model

    def embed(
        self: MatrixFactorization,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input shape: (batch_size, num_features)
        embed = self.embedding(feature_hashes, per_sample_weights=feature_weights)
        # shape: (batch_size, embed_dim)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)
        # shape: (batch_size, embed_dim)
        return embed

    def forward(
        self: MatrixFactorization,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(feature_hashes, feature_weights)

    def score(
        self: MatrixFactorization,
        user_feature_hashes: torch.Tensor,
        item_feature_hashes: torch.Tensor,
        *,
        user_feature_weights: torch.Tensor | None = None,
        item_feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_embed = self(user_feature_hashes, user_feature_weights)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_feature_hashes, item_feature_weights)
        # shape: (batch_size, embed_dim)
        # output shape: (batch_size)
        return (user_embed * item_embed).sum(-1)

    def full_predict(
        self: MatrixFactorization,
        user_feature_hashes: torch.Tensor,
        item_feature_hashes: torch.Tensor,
        *,
        user_feature_weights: torch.Tensor | None = None,
        item_feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_embed = self(user_feature_hashes, per_sample_weights=user_feature_weights)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_feature_hashes, per_sample_weights=item_feature_weights)
        # shape: (batch_size, embed_dim)
        # output shape: (batch_size, batch_size)
        return torch.mm(user_embed, item_embed.T)
