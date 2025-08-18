from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import torch
from sentence_transformers import SentenceTransformer, models
from transformers.models.bert import BertConfig, BertModel

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


def to_sentence_transformer(
    model: PreTrainedModel,
    *,
    tokenizer_name: str = "google-bert/bert-base-uncased",
    pooling_mode: str = "mean",
) -> SentenceTransformer:
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        transformer = models.Transformer(tmpdir, tokenizer_name_or_path=tokenizer_name)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode
        )
        normalize = models.Normalize()

        return SentenceTransformer(
            modules=[transformer, pooling, normalize], device="cpu"
        )


class PoolingTransformer(torch.nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name_or_path: str | None = None,
        hidden_size: int = 384,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 32,
        tokenizer_name: str = "google-bert/bert-base-uncased",
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()

        if model_name_or_path:
            self.model = SentenceTransformer(model_name_or_path)
            return

        model = self.init_model(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
        )
        self.model = to_sentence_transformer(
            model, tokenizer_name=tokenizer_name, pooling_mode=pooling_mode
        )

    def init_model(
        self,
        *,
        hidden_size: int = 384,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 32,
    ) -> BertModel:
        config = BertConfig(
            vocab_size=2,
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
        )
        return BertModel(config)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        attention_mask = (inputs_embeds != 0).any(dim=-1).to(self.model.dtype)
        features = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        return self.model(features)["sentence_embedding"]

    def save(self, save_directory: str) -> None:
        self.model.save(save_directory)
