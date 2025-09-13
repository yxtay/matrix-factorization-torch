from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic
import torch

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.bert import BertModel


class ModelConfig(pydantic.BaseModel):
    vocab_size: int = 30522
    hidden_size: int = 384
    num_hidden_layers: int = 3
    num_attention_heads: int = 12
    intermediate_size: int = 1536
    hidden_act: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    max_position_embeddings: int = 512

    tokenizer_name: str = "google-bert/bert-base-uncased"
    pooling_mode: Literal["mean", "max", "cls", "pooler"] = "mean"


def init_bert(config: ModelConfig) -> BertModel:
    from transformers.models.bert import BertConfig, BertModel

    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
    )
    return BertModel(bert_config)


def to_sentence_transformer(
    model: PreTrainedModel,
    *,
    tokenizer_name: str = "google-bert/bert-base-uncased",
    pooling_mode: str = "mean",
) -> SentenceTransformer:
    import tempfile

    from sentence_transformers import SentenceTransformer, models

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
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = ModelConfig.model_validate(config)

        model = init_bert(self.config)
        self.model = to_sentence_transformer(
            model,
            tokenizer_name=self.config.tokenizer_name,
            pooling_mode=self.config.pooling_mode,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        attention_mask = (inputs_embeds != 0).any(dim=-1).to(self.model.dtype)
        features = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        return self.model(features)["sentence_embedding"]

    def save(self, save_directory: str) -> None:
        self.model.save(save_directory)
