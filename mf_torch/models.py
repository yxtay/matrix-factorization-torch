from __future__ import annotations

from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812
from sentence_transformers import SentenceTransformer


class MatrixFactorization(torch.nn.Module):
    def __init__(
        self: Self,
        *,
        model_name_or_path: str,
        num_hidden_layers: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = self.init_encoder(
            model_name_or_path, num_hidden_layers=num_hidden_layers
        )
        self.normalize = normalize

    @property
    def device(self: Self) -> torch.device:
        return self.encoder.device

    def init_encoder(
        self: Self, model_name_or_path: str, num_hidden_layers: int | None
    ) -> SentenceTransformer:
        config_kwargs = (
            {"num_hidden_layers": num_hidden_layers} if num_hidden_layers else None
        )
        encoder = SentenceTransformer(
            model_name_or_path, device="cpu", config_kwargs=config_kwargs
        )
        # freeze embeddings layer
        for name, module in encoder.named_modules():
            if "embeddings" in name:
                for param in module.parameters():
                    param.requires_grad = False
                break
        return encoder

    def embed(self: Self, text: list[str]) -> torch.Tensor:
        # input shape: (batch_size,)
        tokens = self.encoder.tokenize(text)
        # shape: (batch_size, seq_len)
        embed = self.encoder(tokens)["sentence_embedding"]
        # shape: (batch_size, embed_dim)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)
        # shape: (batch_size, embed_dim)
        return embed

    def forward(self: Self, text: str | list[str]) -> torch.Tensor:
        is_single_example = isinstance(text, str)
        if is_single_example:
            text = [text]

        embed = self.embed(text)

        if is_single_example:
            embed = embed.squeeze(0)
        return embed

    def score(
        self: Self,
        user_text: list[str],
        item_text: list[str],
    ) -> torch.Tensor:
        user_embed = self(user_text)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_text)
        # shape: (batch_size, embed_dim)
        # output shape: (batch_size)
        return (user_embed * item_embed).sum(-1)

    def score_full(
        self: Self,
        user_text: list[str],
        item_text: list[str],
    ) -> torch.Tensor:
        user_embed = self(user_text)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_text)
        # shape: (batch_size, embed_dim)
        # output shape: (batch_size, batch_size)
        return torch.mm(user_embed, item_embed.t())
