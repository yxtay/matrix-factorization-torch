from __future__ import annotations

import pathlib
import random
from typing import TYPE_CHECKING

import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.loggers as lp_loggers
import torch
from lightning import LightningModule
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from mf_torch.data.lightning import BATCH_TYPE
from mf_torch.params import (
    EMBEDDING_DIM,
    EXPORTED_PROGRAM_PATH,
    METRIC,
    NUM_EMBEDDINGS,
    ONNX_PROGRAM_PATH,
    SCRIPT_MODULE_PATH,
    TOP_K,
    USER_ID_COL,
)

if TYPE_CHECKING:
    from typing import Self

    import pandas as pd
    import torchmetrics
    from lightning import Callback, Trainer
    from lightning.pytorch.cli import ArgsType
    from mlflow import MlflowClient

    import mf_torch.models as mf_models
    from mf_torch.data.lightning import (
        ItemsProcessor,
        UsersProcessor,
    )


class MatrixFactorizationLitModule(LightningModule):
    def __init__(  # noqa: PLR0913
        self: Self,
        *,
        num_embeddings: int = NUM_EMBEDDINGS,  # noqa: ARG002
        embedding_dim: int = EMBEDDING_DIM,  # noqa: ARG002
        train_loss: str = "PairwiseHingeLoss",  # noqa: ARG002
        embedder_type: str = "base",  # noqa: ARG002
        num_heads: int = 1,  # noqa: ARG002
        dropout: float = 0.0,  # noqa: ARG002
        normalize: bool = True,  # noqa: ARG002
        hard_negatives_ratio: float | None = None,  # noqa: ARG002
        sigma: float = 1.0,  # noqa: ARG002
        margin: float = 1.0,  # noqa: ARG002
        learning_rate: float = 0.01,  # noqa: ARG002
        top_k: int = TOP_K,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: mf_models.MatrixFactorization | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.metrics: torch.nn.ModuleDict | None = None
        self.users_processor: UsersProcessor | None = None
        self.items_processor: ItemsProcessor | None = None

        supported_embedder = {"base", "attention", "transformer"}
        if self.hparams.embedder_type not in supported_embedder:
            msg = f"{self.hparams.embedder_type = }, not one of {supported_embedder}"
            raise ValueError(msg)

    def forward(
        self: Self,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            msg = "`model` must be initialised first"
            raise ValueError(msg)

        return self.model(
            feature_hashes=feature_hashes, feature_weights=feature_weights
        )

    def recommend(
        self: Self,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor,
        top_k: int = TOP_K,
        user_id: int | None = None,
        exclude_item_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        if self.users_processor is None or self.items_processor is None:
            msg = "`user_processor` and `item_processor` must be initialised first"
            raise ValueError(msg)

        with torch.inference_mode():
            embed = self(
                feature_hashes.unsqueeze(0), feature_weights.unsqueeze(0)
            ).numpy(force=True)

        history = self.users_processor.get_activity(user_id, "history")
        exclude_item_ids = (exclude_item_ids or []) + list(history.keys())

        return self.items_processor.search(
            embed, exclude_item_ids=exclude_item_ids, top_k=top_k
        ).drop(columns="embedding")

    def compute_losses(
        self: Self, batch: BATCH_TYPE, step_name: str = "train"
    ) -> dict[str, torch.Tensor]:
        if self.loss_fns is None:
            msg = "`loss_fns` must be initialised first"
            raise ValueError(msg)

        targets: torch.Tensor = batch["targets"]
        # shape: (batch_size)

        # user
        users: dict[str, torch.Tensor] = batch["users"]
        user_feature_hashes = users["feature_hashes"]
        # shape: (batch_size, num_user_features)
        user_feature_weights = users["feature_weights"]
        # shape: (batch_size, num_user_features)
        user_embed = self(user_feature_hashes, user_feature_weights)
        # shape: (batch_size, embed_dim)

        # item
        items: dict[str, torch.Tensor] = batch["items"]
        item_feature_hashes = items["feature_hashes"]
        # shape: (batch_size, num_item_features)
        item_feature_weights = items["feature_weights"]
        # shape: (batch_size, num_item_features)
        item_embed = self(item_feature_hashes, item_feature_weights)
        # shape: (batch_size, embed_dim)

        n_users, n_items = targets.size()
        nnz = targets.values().numel()
        metrics = {
            "batch/users": n_users,
            "batch/items": n_items,
            "batch/nnz": nnz,
            "batch/sparsity": nnz / targets.numel(),
        }
        losses = {
            f"{step_name}/{loss_fn.__class__.__name__}": loss_fn(
                user_embed=user_embed, item_embed=item_embed, targets=targets
            )
            for loss_fn in self.loss_fns
        }
        return losses | metrics

    def update_metrics(
        self: Self, example: dict[str, torch.Tensor], step_name: str = "train"
    ) -> torchmetrics.MetricCollection:
        import torchmetrics.retrieval as tm_retrieval

        if self.metrics is None:
            msg = "`metrics` must be initialised first"
            raise ValueError(msg)

        user_id = example[USER_ID_COL]
        pred_scores = self.recommend(
            example["feature_hashes"],
            example["feature_weights"],
            top_k=self.hparams.top_k,
            user_id=user_id,
        )

        pred_scores = dict(
            zip(pred_scores["movie_id"], pred_scores["score"], strict=True)
        )
        target_scores = dict(
            zip(example["target"]["movie_id"], example["target"]["rating"], strict=True)
        )

        movie_ids = list(target_scores.keys() | pred_scores.keys())
        preds = [
            pred_scores.get(
                movie_id,
                -random.random(),  # devskim: ignore DS148264 # nosec
            )
            for movie_id in movie_ids
        ]
        preds = torch.as_tensor(preds)
        target = [target_scores.get(movie_id, 0) for movie_id in movie_ids]
        target = torch.as_tensor(target)
        indexes = torch.ones_like(preds, dtype=torch.long) * user_id

        metrics: torchmetrics.MetricCollection = self.metrics[step_name]
        for metric in metrics.values():
            if isinstance(metric, tm_retrieval.RetrievalNormalizedDCG):
                metric.update(preds=preds, target=target, indexes=indexes)
            else:
                metric.update(preds=preds, target=target > 0, indexes=indexes)
        return metrics

    def training_step(self: Self, batch: BATCH_TYPE, _: int) -> torch.Tensor:
        losses = self.compute_losses(batch, step_name="train")
        self.log_dict(losses)
        return losses[f"train/{self.hparams.train_loss}"]

    def validation_step(self: Self, batch: dict[str, torch.Tensor], _: int) -> None:
        metrics = self.update_metrics(batch, step_name="val")
        self.log_dict(metrics)

    def test_step(self: Self, batch: dict[str, torch.Tensor], _: int) -> None:  # noqa: PT019
        metrics = self.update_metrics(batch, step_name="test")
        self.log_dict(metrics)

    def predict_step(
        self: Self, batch: dict[str, torch.Tensor], _: int
    ) -> pd.DataFrame:
        return self.recommend(
            batch["feature_hashes"],
            batch["feature_weights"],
            top_k=self.hparams.top_k,
            user_id=batch[USER_ID_COL],
        )

    def on_train_start(self: Self) -> None:
        if self.metrics is None:
            msg = "`metrics` must be initialised first"
            raise ValueError(msg)

        params = self.hparams | self.trainer.datamodule.hparams
        metrics = {
            key: self.trainer.callback_metrics.get(key, 0.0)
            for key in self.metrics["val"]
        }
        for logger in self.loggers:
            if isinstance(logger, lp_loggers.TensorBoardLogger):
                logger.log_hyperparams(params=params, metrics=metrics)

    def on_validation_start(self: Self) -> None:
        self.users_processor = self.trainer.datamodule.users_processor
        self.users_processor.get_index()
        self.items_processor = self.trainer.datamodule.items_processor
        self.items_processor.get_index(self)

    def on_test_start(self: Self) -> None:
        self.on_validation_start()

    def on_predict_start(self: Self) -> None:
        self.on_validation_start()

    def configure_optimizers(self: Self) -> torch.optim.Optimizer:
        if self.model is None:
            msg = "`model` must be initialised first"
            raise ValueError(msg)

        if self.model.sparse:
            return torch.optim.SparseAdam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self: Self) -> list[Callback]:
        checkpoint = lp_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        early_stop = lp_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"], min_delta=0.001
        )
        return [checkpoint, early_stop]

    def configure_model(self: Self) -> None:
        if self.model is None:
            self.model = self.get_model()
            # self.compile()
        if self.loss_fns is None:
            self.loss_fns = self.get_loss_fns()
        if self.metrics is None:
            self.metrics = self.get_metrics()

    def get_model(self: Self) -> torch.nn.Module:
        import mf_torch.models as mf_models

        match self.hparams.embedder_type:
            case "base":
                embedder = mf_models.EmbeddingBag(
                    num_embeddings=self.hparams.num_embeddings,
                    embedding_dim=self.hparams.embedding_dim,
                )
            case "attention":
                embedder = mf_models.AttentionEmbeddingBag(
                    num_embeddings=self.hparams.num_embeddings,
                    embedding_dim=self.hparams.embedding_dim,
                    num_heads=self.hparams.num_heads,
                    dropout=self.hparams.dropout,
                )
            case "transformer":
                embedder = mf_models.TransformerEmbeddingBag(
                    num_embeddings=self.hparams.num_embeddings,
                    embedding_dim=self.hparams.embedding_dim,
                    num_heads=self.hparams.num_heads,
                    dropout=self.hparams.dropout,
                )
            case _:
                msg = f"{self.hparams.embedder_type = }"
                raise NotImplementedError(msg)

        return mf_models.MatrixFactorization(
            embedder=embedder, normalize=self.hparams.normalize
        )

    def get_loss_fns(self: Self) -> torch.nn.ModuleList:
        import mf_torch.losses as mf_losses

        loss_classes = [
            mf_losses.AlignmentLoss,
            mf_losses.ContrastiveLoss,
            mf_losses.AlignmentContrastiveLoss,
            mf_losses.UniformityLoss,
            mf_losses.AlignmentUniformityLoss,
            mf_losses.InfomationNoiseContrastiveEstimationLoss,
            mf_losses.MutualInformationNeuralEstimationLoss,
            mf_losses.PairwiseHingeLoss,
            mf_losses.PairwiseLogisticLoss,
        ]
        loss_fns = [
            loss_class(
                hard_negatives_ratio=self.hparams.get("hard_negatives_ratio"),
                sigma=self.hparams.sigma,
                margin=self.hparams.margin,
            )
            for loss_class in loss_classes
        ]
        return torch.nn.ModuleList(loss_fns)

    def get_metrics(self: Self) -> torch.nn.ModuleDict:
        import torchmetrics
        import torchmetrics.retrieval as tm_retrieval

        top_k = self.hparams.top_k
        metrics = {
            step_name: torchmetrics.MetricCollection(
                tm_retrieval.RetrievalNormalizedDCG(top_k=top_k),
                tm_retrieval.RetrievalRecall(top_k=top_k),
                tm_retrieval.RetrievalPrecision(top_k=top_k),
                tm_retrieval.RetrievalMAP(top_k=top_k),
                tm_retrieval.RetrievalHitRate(top_k=top_k),
                tm_retrieval.RetrievalMRR(top_k=top_k),
                prefix=f"{step_name}/",
            )
            for step_name in ["val", "test"]
        }
        return torch.nn.ModuleDict(metrics)

    @property
    def example_input_array(self: Self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((2, 2), dtype=torch.int, device=self.device),
            torch.zeros((2, 2), dtype=self.dtype, device=self.device),
        )

    def export_torchscript(
        self: Self, path: str | None = None
    ) -> torch.jit.ScriptModule:
        script_module = torch.jit.script(self.model.eval())  # devskim: ignore DS189424

        if path is None:
            path = pathlib.Path(self.trainer.log_dir) / SCRIPT_MODULE_PATH
        torch.jit.save(script_module, path)  # nosec
        return script_module

    def export_dynamo(
        self: Self, path: str | None = None
    ) -> torch.export.ExportedProgram:
        batch = torch.export.Dim("batch")
        features = torch.export.Dim("features")
        dynamic_shapes = {
            "feature_hashes": (batch, features),
            "feature_weights": (batch, features),
        }
        exported_program = torch.export.export(
            self.model.eval(),
            self.example_input_array,
            dynamic_shapes=dynamic_shapes,
        )

        if path is None:
            path = pathlib.Path(self.trainer.log_dir) / EXPORTED_PROGRAM_PATH
        torch.export.save(exported_program, path)  # nosec
        return exported_program

    def export_onnx(self: Self, path: str | None = None) -> torch.onnx.ONNXProgram:
        if path is None:
            path = pathlib.Path(self.trainer.log_dir) / ONNX_PROGRAM_PATH

        dynamo_path = pathlib.Path(path).parent / EXPORTED_PROGRAM_PATH
        exported_program = self.export_dynamo(dynamo_path)
        return torch.onnx.export(
            exported_program,
            self.example_input_array,
            path,
            dynamo=True,
            optimize=True,
            verify=True,
        )


class LoggerSaveConfigCallback(SaveConfigCallback):
    @rank_zero_only
    def save_config(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        import tempfile

        for logger in trainer.loggers:
            if isinstance(logger, lp_loggers.MLFlowLogger):
                with tempfile.TemporaryDirectory() as path:
                    config_path = pathlib.Path(path) / self.config_filename
                    self.parser.save(
                        self.config,
                        config_path,
                        skip_none=False,
                        overwrite=self.overwrite,
                        multifile=self.multifile,
                    )
                    mlflow_client: MlflowClient = logger.experiment
                    mlflow_client.log_artifact(
                        run_id=logger.run_id, local_path=config_path
                    )


def time_now_isoformat() -> str:
    import datetime

    datetime_now = datetime.datetime.now(datetime.UTC).astimezone()
    return datetime_now.isoformat(timespec="seconds")


def cli_main(
    args: ArgsType = None,
    *,
    run: bool = True,
    experiment_name: str = time_now_isoformat(),
    run_name: str | None = None,
) -> LightningCLI:
    from jsonargparse import lazy_instance

    from mf_torch.data.lightning import MatrixFactorizationDataModule
    from mf_torch.params import MLFLOW_DIR, TENSORBOARD_DIR

    run_name = run_name or time_now_isoformat()
    tensorboard_logger = {
        "class_path": "TensorBoardLogger",
        "init_args": {
            "save_dir": TENSORBOARD_DIR,
            "name": experiment_name,
            "version": run_name,
            # "log_graph": True,
            "default_hp_metric": False,
        },
    }
    mlflow_logger = {
        "class_path": "MLFlowLogger",
        "init_args": {
            "save_dir": MLFLOW_DIR,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "log_model": True,
        },
    }
    progress_bar = lazy_instance(lp_callbacks.RichProgressBar)
    trainer_defaults = {
        "accelerator": "cpu",
        "precision": "bf16-mixed",
        "logger": [tensorboard_logger, mlflow_logger],
        "callbacks": [progress_bar],
        "max_epochs": 1,
        "max_time": "00:02:00:00",
        "num_sanity_val_steps": 0,
    }
    return LightningCLI(
        MatrixFactorizationLitModule,
        MatrixFactorizationDataModule,
        save_config_callback=LoggerSaveConfigCallback,
        trainer_defaults=trainer_defaults,
        args=args,
        run=run,
    )


if __name__ == "__main__":
    import rich

    from mf_torch.data.lightning import MatrixFactorizationDataModule

    datamodule = MatrixFactorizationDataModule()
    datamodule.prepare_data()
    datamodule.setup("fit")
    model = MatrixFactorizationLitModule()
    model.configure_model()

    rich.print(model(*model.example_input_array))
    rich.print(model.compute_losses(next(iter(datamodule.train_dataloader()))))

    trainer_args = {
        "fast_dev_run": True,
        # "max_epochs": -1,
        # "overfit_batches": 1,
    }
    cli_main(args={"fit": {"trainer": trainer_args}})
