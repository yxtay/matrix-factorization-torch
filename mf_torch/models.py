from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer


class MatrixFactorization(torch.nn.Module):
    def __init__(
        self,
        *,
        model_name_or_path: str,
        num_hidden_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = self.init_encoder(
            model_name_or_path, num_hidden_layers=num_hidden_layers
        )

    def init_encoder(
        self, model_name_or_path: str, num_hidden_layers: int | None
    ) -> SentenceTransformer:
        config_kwargs = {}
        if num_hidden_layers:
            config_kwargs["num_hidden_layers"] = num_hidden_layers

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

    def forward(self, text: list[str]) -> torch.Tensor:
        # input shape: (batch_size,)
        tokens = self.encoder.tokenize(text)
        # shape: (batch_size, seq_len)
        # output shape: (batch_size, embed_dim)
        return self.encoder(tokens)["sentence_embedding"]

    def save_pretrained(self, path: str) -> None:
        self.encoder.save_pretrained(path)
