from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

import torch

from mf_torch.params import MODEL_NAME

if TYPE_CHECKING:
    import bentoml
    from lightning import Trainer

    from mf_torch.lightning import MatrixFactorizationLitModule


def load_args(ckpt_path: str | None) -> dict[str, Any]:
    if not ckpt_path:
        return {"model": {}, "data": {}}

    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # nosec
    model_args = checkpoint["hyper_parameters"]
    model_args = {
        key: value for key, value in model_args.items() if not key.startswith("_")
    }

    data_args = checkpoint["datamodule_hyper_parameters"]
    data_args = {
        key: value for key, value in data_args.items() if not key.startswith("_")
    }
    return {"model": model_args, "data": data_args}


def prepare_trainer(
    ckpt_path: str | None = None, stage: str = "validate", fast_dev_run: int = 0
) -> Trainer:
    from mf_torch.lightning import cli_main

    if ckpt_path is None:
        args = {"trainer": {"accelerator": "cpu", "fast_dev_run": True}}
        return cli_main({"fit": args}).trainer

    with tempfile.TemporaryDirectory() as tmp:
        trainer_args = {
            "logger": False,
            "fast_dev_run": fast_dev_run,
            "enable_checkpointing": False,
            "default_root_dir": tmp,
        }
        args = {"trainer": trainer_args, "ckpt_path": ckpt_path, **load_args(ckpt_path)}
        return cli_main({stage: args}).trainer


def save_model(trainer: Trainer) -> None:
    import bentoml

    with bentoml.models.create(MODEL_NAME) as model_ref:
        model: MatrixFactorizationLitModule = trainer.model
        model.save(model_ref.path)


def test_bento(
    service: type[bentoml.Service], api_name: str, api_input: dict[str, Any]
) -> dict[str, Any]:
    from starlette.testclient import TestClient

    # disable prometheus, which can cause duplicated metrics error with repeated runs
    service.config["metrics"] = {"enabled": False}

    asgi_app = service.to_asgi()
    with TestClient(asgi_app) as client:
        response = client.post(f"/{api_name}", json=api_input)
        response.raise_for_status()
        return response.json()


def test_queries() -> None:
    import pydantic
    import rich

    from mf_torch.bentoml.service import (
        EXAMPLE_ITEM,
        EXAMPLE_USER,
        ItemCandidate,
        ItemQuery,
        Service,
        UserQuery,
    )

    example_item_data = test_bento(Service, "item_id", {"item_id": 1})
    example_item = ItemQuery.model_validate(example_item_data)
    rich.print(example_item)
    if example_item != EXAMPLE_ITEM:
        msg = f"{example_item = } != {EXAMPLE_ITEM = }"
        raise ValueError(msg)

    example_user_data = test_bento(Service, "user_id", {"user_id": 1})
    example_user = UserQuery.model_validate(example_user_data)
    rich.print(example_user)
    exclude_fields = {"history", "target"}
    if example_user.model_dump(exclude=exclude_fields) != EXAMPLE_USER.model_dump(
        exclude=exclude_fields
    ):
        msg = f"{example_user = } != {EXAMPLE_USER = }"
        raise ValueError(msg)

    item_recs = test_bento(Service, "recommend_with_item_id", {"item_id": 1})
    item_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(item_recs)
    rich.print(item_recs)

    user_recs = test_bento(Service, "recommend_with_user_id", {"user_id": 1})
    user_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(user_recs)
    rich.print(user_recs)


def main(ckpt_path: str | None = None) -> None:
    trainer = prepare_trainer(ckpt_path)
    save_model(trainer=trainer)
    test_queries()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
