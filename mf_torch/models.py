from __future__ import annotations

from typing import Self

import torch
import torch.nn.functional as F

from mf_torch.params import PADDING_IDX


class MatrixFactorization(torch.nn.Module):
    def __init__(
        self: Self, embedder: torch.nn.Module, *, normalize: bool = True
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.normalize = normalize

    @property
    def sparse(self: Self) -> bool:
        return self.embedder.sparse

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

    @torch.jit.export
    def score(
        self: Self,
        user_feature_hashes: torch.Tensor,
        item_feature_hashes: torch.Tensor,
        user_feature_weights: torch.Tensor | None = None,
        item_feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_embed = self(user_feature_hashes, user_feature_weights)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_feature_hashes, item_feature_weights)
        # shape: (batch_size, embed_dim)
        # output shape: (batch_size)
        return (user_embed * item_embed).sum(-1)

    @torch.jit.export
    def score_full(
        self: Self,
        user_feature_hashes: torch.Tensor,
        item_feature_hashes: torch.Tensor,
        user_feature_weights: torch.Tensor | None = None,
        item_feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_embed = self(user_feature_hashes, feature_weights=user_feature_weights)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_feature_hashes, feature_weights=item_feature_weights)
        # shape: (batch_size, embed_dim)
        # output shape: (batch_size, batch_size)
        return torch.mm(user_embed, item_embed.T)


class EmbeddingBag(torch.nn.Module):
    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        max_norm: float | None = None,
        norm_type: float = 2.0,
    ) -> None:
        super().__init__()
        self.embedder = torch.nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=PADDING_IDX,
            max_norm=max_norm,
            norm_type=norm_type,
            mode="sum",
            sparse=True,
        )

    @property
    def sparse(self: Self) -> bool:
        return self.embedder.sparse

    def forward(
        self: Self, hashes: torch.Tensor, per_sample_weights: torch.Tensor | None
    ) -> torch.Tensor:
        return self.embedder(hashes, per_sample_weights=per_sample_weights)


class AttentionEmbeddingBag(torch.nn.Module):
    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedder = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=PADDING_IDX,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        self.encoder = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            # use batch_first=False due to issue with tracing fastpath
            batch_first=False,
        )

    @property
    def sparse(self: Self) -> bool:
        return self.embedder.sparse

    def forward(
        self: Self, hashes: torch.Tensor, _: torch.Tensor | None
    ) -> torch.Tensor:
        mask = hashes != PADDING_IDX
        # shape: (batch_size, num_features)
        denominator = mask.sum(dim=-1, keepdim=True)
        # shape: (batch_size, 1)

        embedded = self.embedder(hashes)
        # shape: (batch_size, num_features, embed_dim)
        # transpose so that dimensions is (seq_len, batch_size, embed_dim)
        embedded = embedded.transpose(0, 1)
        encoded, _ = self.encoder(
            query=embedded,
            key=embedded,
            value=embedded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        encoded = encoded.transpose(0, 1)
        # shape: (batch_size, num_features, embed_dim)
        # output: (batch_size, embed_dim)
        return (encoded * mask[:, :, None] / denominator[:, :, None]).sum(dim=-2)


class TransformerEmbeddingBag(torch.nn.Module):
    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedder = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=PADDING_IDX,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            # use batch_first=False due to issue with tracing fastpath
            batch_first=False,
        )

    @property
    def sparse(self: Self) -> bool:
        return self.embedder.sparse

    def forward(
        self: Self, hashes: torch.Tensor, _: torch.Tensor | None
    ) -> torch.Tensor:
        mask = hashes != PADDING_IDX
        # shape: (batch_size, num_features)
        denominator = mask.sum(dim=-1, keepdim=True)
        # shape: (batch_size, 1)

        embedded = self.embedder(hashes)
        # shape: (batch_size, num_features, embed_dim)
        # transpose so that dimensions is (seq_len, batch_size, embed_dim)
        embedded = embedded.transpose(0, 1)
        encoded = self.encoder(embedded, src_key_padding_mask=~mask)
        encoded = encoded.transpose(0, 1)
        # shape: (batch_size, num_features, embed_dim)
        # output: (batch_size, embed_dim)
        return (encoded * mask[:, :, None] / denominator[:, :, None]).sum(dim=-2)
