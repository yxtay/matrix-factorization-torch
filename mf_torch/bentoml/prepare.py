from __future__ import annotations

from typing import TypeVar

import torch
from lightning import Trainer
from pydantic import BaseModel

from mf_torch.bentoml.schemas import ItemCandidate, UserQuery
from mf_torch.data.lightning import MatrixFactorizationDataModule
from mf_torch.lightning import MatrixFactorizationLitModule
from mf_torch.params import (
    CHECKPOINT_PATH,
    EXPORTED_PROGRAM_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
    USERS_TABLE_NAME,
)

T = TypeVar("T", bound=BaseModel)


def load_args(ckpt_path: str | None) -> dict:
    if not ckpt_path:
        return {"model": {}, "data": {}}

    checkpoint = torch.load(
        ckpt_path, weights_only=False, map_location=torch.device("cpu")
    )
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
    if ckpt_path:
        datamodule = MatrixFactorizationDataModule.load_from_checkpoint(ckpt_path)
        model = MatrixFactorizationLitModule.load_from_checkpoint(ckpt_path)
    else:
        datamodule = MatrixFactorizationDataModule()
        model = MatrixFactorizationLitModule()

    trainer = Trainer(logger=False, enable_checkpointing=False)
    trainer.validate(model=model, datamodule=datamodule)
    return trainer


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
    save_model(trainer=trainer)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
