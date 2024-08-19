from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from docarray import DocList

from mf_torch.bentoml.schemas import ItemCandidate, Query
from mf_torch.params import (
    CHECKPOINT_PATH,
    EXPORTED_PROGRAM_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
    PADDING_IDX,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import lancedb.table
    import pandas as pd
    from lightning import Trainer

    from mf_torch.lightning import MatrixFactorizationLitModule


def load_args(ckpt_path: str | None) -> dict:
    if not ckpt_path:
        return {"model": {}, "data": {}}

    checkpoint = torch.load(ckpt_path)
    model_args = checkpoint["hyper_parameters"]
    model_args = {
        key: value for key, value in model_args.items() if not key.startswith("_")
    }

    data_args = checkpoint["datamodule_hyper_parameters"]
    data_args = {
        key: value for key, value in data_args.items() if not key.startswith("_")
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


def prepare_items(trainer: Trainer) -> Iterable[pd.DataFrame]:
    datamodule = trainer.datamodule
    return (
        trainer.datamodule.get_items_dataset(prefix="")
        .map(ItemCandidate.model_validate)
        .batch(datamodule.hparams.batch_size)
        .map(DocList[ItemCandidate])
        .map(partial(embed_queries, model=trainer.model.original_model))
        .map(DocList[ItemCandidate].to_dataframe)
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

        embeddings = model(feature_hashes, feature_weights)
        queries.embedding = list(embeddings)
    return queries


def index_items(
    items: Iterable[pd.DataFrame], lance_db_path: str = LANCE_DB_PATH
) -> lancedb.table.LanceTable:
    import datetime

    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    iterator = iter(items)
    batch = next(iterator)

    num_items = len(items) * len(batch)
    embedding_dim = batch.iloc[0, batch.columns.get_loc("embedding")].shape[-1]

    num_partitions = int(num_items**0.5)
    num_sub_vectors = embedding_dim // 8

    class ItemSchema(ItemCandidate, LanceModel):
        feature_hashes: list[int]
        feature_weights: list[float]
        embedding: Vector(embedding_dim)

    db = lancedb.connect(lance_db_path)
    table = db.create_table(
        ITEMS_TABLE_NAME,
        data=batch,
        schema=ItemSchema,
        mode="overwrite",
    )
    table.add(iterator)
    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        vector_column_name="embedding",
    )
    table.compact_files()
    table.cleanup_old_versions(datetime.timedelta(days=1))
    return table


def save_model(trainer: Trainer) -> None:
    import shutil

    import bentoml

    with bentoml.models.create(MODEL_NAME) as model_ref:
        shutil.copytree(LANCE_DB_PATH, model_ref.path_of(LANCE_DB_PATH))
        trainer.save_checkpoint(model_ref.path_of(CHECKPOINT_PATH))
        trainer.model.export_dynamo(model_ref.path_of(EXPORTED_PROGRAM_PATH))


def load_indexed_items() -> DocList[ItemCandidate]:
    import bentoml
    import lancedb
    from pydantic import TypeAdapter

    lancedb_path = bentoml.models.get(MODEL_NAME).path_of(LANCE_DB_PATH)
    tbl = lancedb.connect(lancedb_path).open_table(ITEMS_TABLE_NAME)
    return TypeAdapter(list[ItemCandidate]).validate_python(tbl.to_arrow().to_pylist())


def main(ckpt_path: str | None) -> None:
    trainer = prepare_trainer(ckpt_path)
    items = prepare_items(trainer)
    index_items(items=items)
    save_model(trainer=trainer)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
