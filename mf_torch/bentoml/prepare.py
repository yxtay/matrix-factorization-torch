from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from docarray import DocList

from mf_torch.bentoml.schemas import ItemCandidate, Query
from mf_torch.params import (
    CHECKPOINT_PATH,
    ITEMS_DOC_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
    PADDING_IDX,
    SCRIPTMODULE_PATH,
)

if TYPE_CHECKING:
    from typing import Self

    import lancedb.table
    from lightning import Trainer

    from mf_torch.lightning import MatrixFactorizationLitModule


def load_args(ckpt_path: str | None) -> dict:
    from mf_torch.data.lightning import MatrixFactorizationDataModule
    from mf_torch.lightning import MatrixFactorizationLitModule

    if not ckpt_path:
        return {}

    checkpoint = torch.load(ckpt_path)
    model_args = checkpoint["hyper_parameters"]
    model_args = {
        key: value
        for key, value in model_args.items()
        if key in inspect.signature(MatrixFactorizationLitModule.__init__).parameters
    }

    data_args = checkpoint["datamodule_hyper_parameters"]
    data_args = {
        key: value
        for key, value in data_args.items()
        if key in inspect.signature(MatrixFactorizationDataModule.__init__).parameters
    }
    return {"model": model_args, "data": data_args}


def prepare_trainer(ckpt_path: str | None = None) -> Trainer:
    import tempfile

    from mf_torch.lightning import cli_main

    with tempfile.TemporaryDirectory() as tmp:
        trainer_args = {
            "precision": "bf16-mixed",
            "num_sanity_val_steps": 0,
        }
        if ckpt_path is not None:
            trainer_args["logger"] = False
            trainer_args["enable_checkpointing"] = False
            trainer_args["default_root_dir"] = tmp
        args = {
            "fit": {
                "trainer": trainer_args,
                "ckpt_path": ckpt_path,
                **load_args(ckpt_path),
            }
        }
        cli = cli_main(args)
    return cli.trainer


def prepare_items(ckpt_path: str | None = None) -> DocList[ItemCandidate]:
    from mf_torch.data.lightning import MatrixFactorizationDataModule

    data_args = load_args(ckpt_path)["data"]
    datamodule = MatrixFactorizationDataModule(**data_args)
    items_dataset = datamodule.get_items_dataset(prefix="")
    return DocList[ItemCandidate](
        ItemCandidate.model_validate(item) for item in items_dataset
    )


def embed_queries(
    queries: DocList[Query], model: MatrixFactorizationLitModule
) -> DocList[ItemCandidate]:
    with torch.inference_mode():
        feature_hashes = torch.nested.nested_tensor(
            queries.feature_hashes
        ).to_padded_tensor(padding=PADDING_IDX)
        feature_weights = torch.nested.nested_tensor(
            queries.feature_weights
        ).to_padded_tensor(padding=PADDING_IDX)

        embeddings = model(feature_hashes, feature_weights).float()
        queries.embedding = list(embeddings)
    return queries


def index_items(
    items: DocList[ItemCandidate], lance_db_path: str = LANCE_DB_PATH
) -> lancedb.table.LanceTable:
    import datetime

    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    embedding_dim = items[0].embedding.size
    num_partitions = int(len(items) ** 0.5)
    num_sub_vectors = embedding_dim // 8

    class ItemSchema(LanceModel):
        movie_id: int
        title: str
        genres: list[str]
        feature_values: list[str]
        feature_hashes: list[float]
        feature_weights: list[float]
        embedding: Vector(embedding_dim)

        @property
        def id(self: Self) -> str:
            return str(self.movie_id)

    db = lancedb.connect(lance_db_path)
    table = db.create_table(
        ITEMS_TABLE_NAME,
        data=items.to_dataframe(),
        schema=ItemSchema,
        mode="overwrite",
    )
    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        vector_column_name="embedding",
    )
    table.compact_files()
    table.cleanup_old_versions(datetime.timedelta(days=1))
    return table


def save_model(items: DocList[ItemCandidate], trainer: Trainer) -> None:
    import shutil

    import bentoml

    with bentoml.models.create(MODEL_NAME) as model_ref:
        shutil.copytree(LANCE_DB_PATH, model_ref.path_of(LANCE_DB_PATH))
        items.push(Path(model_ref.path_of(ITEMS_DOC_PATH)).as_uri())
        trainer.save_checkpoint(model_ref.path_of(CHECKPOINT_PATH))
        trainer.model.save_torchscript(model_ref.path_of(SCRIPTMODULE_PATH))


def load_embedded_items() -> DocList[ItemCandidate]:
    import bentoml

    model_ref = bentoml.models.get(MODEL_NAME)
    path = Path(model_ref.path_of(ITEMS_DOC_PATH)).as_uri()
    return DocList[ItemCandidate].pull(path)


def main(ckpt_path: str | None) -> None:
    trainer = prepare_trainer(ckpt_path)
    items = prepare_items(ckpt_path)
    items = embed_queries(queries=items, model=trainer.model)
    index_items(items=items)
    save_model(items=items, trainer=trainer)


if __name__ == "__main__":
    main()
