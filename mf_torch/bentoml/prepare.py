from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, TypeVar

import torch
from pydantic import BaseModel

from mf_torch.bentoml.schemas import ItemCandidate, Query, UserQuery
from mf_torch.data.load import select_fields
from mf_torch.params import (
    CHECKPOINT_PATH,
    EXPORTED_PROGRAM_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
    USERS_TABLE_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import lancedb.table
    from lightning import Trainer

    from mf_torch.data.lightning import MatrixFactorizationDataModule
    from mf_torch.lightning import MatrixFactorizationLitModule


T = TypeVar("T", bound=BaseModel)


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


def prepare_items(trainer: Trainer) -> Iterable[list[dict]]:
    datamodule: MatrixFactorizationDataModule = trainer.datamodule
    return (
        datamodule.get_items_dataset(prefix="")
        .map(ItemCandidate.model_validate)
        .map(partial(embed_query, model=trainer.model.model))
        .map(ItemCandidate.model_dump)
        .batch(datamodule.hparams.batch_size)
    )


def embed_query(query: Query, model: MatrixFactorizationLitModule) -> ItemCandidate:
    with torch.inference_mode():
        feature_hashes = torch.as_tensor(query.feature_hashes).unsqueeze(0)
        feature_weights = torch.as_tensor(query.feature_weights).unsqueeze(0)
        query.embedding = model(feature_hashes, feature_weights).squeeze(0).numpy()
    return query


def index_items(
    items: Iterable[list[dict]], lance_db_path: str = LANCE_DB_PATH
) -> lancedb.table.LanceTable:
    import datetime

    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    batch = next(iter(items))
    num_items = len(items) * len(batch)
    (embedding_dim,) = batch[0]["embedding"].shape

    num_partitions = int(num_items**0.5)
    num_sub_vectors = embedding_dim // 8

    class ItemSchema(ItemCandidate, LanceModel):
        feature_hashes: list[int]
        feature_weights: list[float]
        embedding: Vector(embedding_dim)

    db = lancedb.connect(lance_db_path)
    table = db.create_table(
        ITEMS_TABLE_NAME,
        data=iter(items),
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


def prepare_users(trainer: Trainer) -> list[dict]:
    datamodule: MatrixFactorizationDataModule = trainer.datamodule

    interactions: Iterable[dict] = datamodule.get_raw_data(subset="train").map(
        partial(
            select_fields,
            fields=["user_id", "gender", "age", "occupation", "zipcode", "movie_id"],
        )
    )
    interacted: dict[int, dict] = {}
    for row in interactions:
        user_id = row["user_id"]
        movie_id = row.pop("movie_id")

        if user_id in interacted:
            curr = interacted[user_id]
            curr.update(row)
            curr["movie_ids"].add(movie_id)
        else:
            curr = row
            curr["movie_ids"] = {movie_id}

        interacted[user_id] = curr
    return list(interacted.values())


def index_users(
    users: Iterable[dict], lance_db_path: str = LANCE_DB_PATH
) -> lancedb.table.LanceTable:
    import datetime

    import lancedb
    from lancedb.pydantic import LanceModel

    class UserSchema(UserQuery, LanceModel):
        pass

    db = lancedb.connect(lance_db_path)
    table = db.create_table(
        USERS_TABLE_NAME,
        data=users,
        schema=UserSchema,
        mode="overwrite",
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


def load_lancedb_indexed(table_name: str, schema: T) -> T:
    import bentoml
    import lancedb
    from pydantic import TypeAdapter

    lancedb_path = bentoml.models.get(MODEL_NAME).path_of(LANCE_DB_PATH)
    tbl = lancedb.connect(lancedb_path).open_table(table_name)
    return TypeAdapter(list[schema]).validate_python(tbl.to_arrow().to_pylist())


def load_indexed_items() -> list[ItemCandidate]:
    return load_lancedb_indexed(table_name=ITEMS_TABLE_NAME, schema=ItemCandidate)


def load_indexed_users() -> list[UserQuery]:
    return load_lancedb_indexed(table_name=USERS_TABLE_NAME, schema=UserQuery)


def main(ckpt_path: str | None) -> None:
    trainer = prepare_trainer(ckpt_path)
    items = prepare_items(trainer)
    index_items(items=items)
    users = prepare_users(trainer)
    index_users(users=users)
    save_model(trainer=trainer)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
