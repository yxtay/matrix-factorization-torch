from __future__ import annotations

from typing import Self

import torch
import torch.nn.functional as F


class MatrixFactorization(torch.nn.Module):
    def __init__(
        self: Self, embedder: torch.nn.Module, *, normalize: bool = True
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.normalize = normalize

    def embed(
        self: Self,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input shape: (batch_size, num_features)
        embed = self.embedder(feature_hashes, per_sample_weights=feature_weights)
        # shape: (batch_size, embed_dim)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)
        # shape: (batch_size, embed_dim)
        return embed

    def forward(
        self: Self,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(feature_hashes, feature_weights)

    def score(
        self: Self,
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
        self: Self,
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


class AttentionEmbeddingBag(torch.nn.Module):
    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int = 0,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        sparse: bool = False,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedder = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            sparse=sparse,
        )
        self.encoder = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self: Self, hashes: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        mask = hashes != 0
        # shape: (batch_size, num_features)

        embedded = self.embedder(hashes)
        # shape: (batch_size, num_features, embed_dim)
        encoded, _ = self.encoder(
            embedded, embedded, embedded, key_padding_mask=~mask, need_weights=False
        )
        # shape: (batch_size, num_features, embed_dim)
        # output: (batch_size, embed_dim)
        return (encoded * mask[:, :, None]).sum(dim=-2)


class TransformerEmbeddingBag(torch.nn.Module):
    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int = 0,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        sparse: bool = False,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedder = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            sparse=sparse,
        )
        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self: Self, hashes: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        mask = hashes != 0
        # shape: (batch_size, num_features)

        embedded = self.embedder(hashes)
        # shape: (batch_size, num_features, embed_dim)
        encoded = self.encoder(embedded, src_key_padding_mask=~mask)
        # shape: (batch_size, num_features, embed_dim)
        # output: (batch_size, embed_dim)
        return (encoded * mask[:, :, None]).sum(dim=-2)
