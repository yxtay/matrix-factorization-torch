from __future__ import annotations

import datetime
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from lightning import LightningModule

from mf_torch.params import EMBEDDING_DIM, NUM_EMBEDDINGS

if TYPE_CHECKING:
    from typing import Self

    import torchmetrics
    from lightning import Callback
    from lightning.pytorch.cli import ArgsType, LightningCLI


METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}
EXPERIMENT_NAME = (
    datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
)


class MatrixFactorizationLitModule(LightningModule):
    def __init__(
        self: Self,
        *,
        num_embeddings: int = NUM_EMBEDDINGS,
        embedding_dim: int = EMBEDDING_DIM,
        train_loss: str = "PairwiseHingeLoss",
        max_norm: float | None = None,
        norm_type: float = 2.0,
        embedder_type: str | None = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        normalize: bool = True,
        hard_negatives_ratio: float | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.metrics = None

        supported_embedder = {None, "attention", "transformer"}
        if (
            self.hparams.get("embedder_type")
            and self.hparams.embedder_type not in supported_embedder
        ):
            msg = (
                f"only {supported_embedder} supported: {self.hparams.embedder_type = }"
            )
            raise ValueError(msg)

    def forward(
        self: Self,
        feature_hashes: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            feature_hashes=feature_hashes, feature_weights=feature_weights
        )

    def score(self: Self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_feature_hashes = batch["user_feature_hashes"]
        # shape: (batch_size, num_user_features)
        item_feature_hashes = batch["item_feature_hashes"]
        # shape: (batch_size, num_user_features)
        user_feature_weights = batch["user_feature_weights"]
        # shape: (batch_size, num_item_features)
        item_feature_weights = batch["item_feature_weights"]
        # shape: (batch_size, num_item_features)
        # output shape: (batch_size)
        return self.model.score(
            user_feature_hashes,
            item_feature_hashes,
            user_feature_weights=user_feature_weights,
            item_feature_weights=item_feature_weights,
        )

    def compute_losses(
        self: Self,
        batch: dict[str, torch.Tensor],
        step_name: str = "train",
    ) -> dict[str, torch.Tensor]:
        label = batch["label"]
        # shape: (batch_size)
        sample_weight = batch["weight"]
        # shape: (batch_size)

        # user
        user_idx = batch["user_idx"]
        # shape: (batch_size)
        user_feature_hashes = batch["user_feature_hashes"]
        # shape: (batch_size, num_user_features)
        user_feature_weights = batch.get("user_feature_weights")
        # shape: (batch_size, num_user_features)

        # item
        item_idx = batch["item_idx"]
        # shape: (batch_size)
        item_feature_hashes = batch["item_feature_hashes"]
        # shape: (batch_size, num_item_features)
        item_feature_weights = batch.get("item_feature_weights")
        # shape: (batch_size, num_item_features)

        # negative item
        neg_item_idx = batch.get("neg_item_idx")
        # shape: (batch_size, neg_multiple)
        neg_item_feature_hashes = batch.get("neg_item_feature_hashes")
        # shape: (batch_size, neg_multiple, num_item_features)
        neg_item_feature_weights = batch.get("neg_item_feature_weights")
        # shape: (batch_size, neg_multiple, num_item_features)

        user_embed = self(user_feature_hashes, user_feature_weights)
        # shape: (batch_size, embed_dim)
        item_embed = self(item_feature_hashes, item_feature_weights)
        # shape: (batch_size, embed_dim)
        if neg_item_feature_hashes is not None:
            num_item_features = neg_item_feature_hashes.size(-1)
            # shape: (batch_size * neg_multiple)
            neg_item_feature_hashes = neg_item_feature_hashes.reshape(
                -1, num_item_features
            )
            # shape: (batch_size * neg_multiple, num_item_features)
            neg_item_feature_weights = neg_item_feature_weights.reshape(
                -1, num_item_features
            )
            # shape: (batch_size * neg_multiple, num_item_features)
            neg_item_embed = self(neg_item_feature_hashes, neg_item_feature_weights)
            # shape: (batch_size * neg_multiple, embed_dim)
            item_embed = torch.cat([item_embed, neg_item_embed])
            # shape: (batch_size * (1 + neg_multiple), embed_dim)
            item_idx = torch.cat([item_idx, neg_item_idx.reshape(-1)])
            # shape: (batch_size * (1 + neg_multiple))

        losses = {}
        for loss_fn in self.loss_fns:
            key = f"{step_name}/{loss_fn.__class__.__name__}"
            losses[key] = loss_fn(
                user_embed=user_embed,
                item_embed=item_embed,
                label=label,
                sample_weight=sample_weight,
                user_idx=user_idx,
                item_idx=item_idx,
            )

        return losses

    def update_metrics(
        self: Self,
        batch: dict[str, torch.Tensor],
        step_name: str = "train",
    ) -> torchmetrics.MetricCollection:
        import torchmetrics.retrieval as tm_retrieval

        user_idx = batch["user_idx"].long()
        label = batch["label"]
        score = self.score(batch)

        metrics: torchmetrics.MetricCollection = self.metrics[step_name]
        for metric in metrics.values():
            if not isinstance(metric, tm_retrieval.RetrievalNormalizedDCG):
                label = label > 0
            metric.update(preds=score, target=label, indexes=user_idx)
        return metrics

    def on_fit_start(self: Self) -> None:
        import lightning.pytorch.loggers as lp_loggers

        if self.global_rank == 0:
            params = {**self.hparams, **self.trainer.datamodule.hparams}
            metrics = {key.replace("val/", "hp/"): 0.0 for key in self.metrics["val"]}
            for logger in self.loggers:
                if isinstance(logger, lp_loggers.TensorBoardLogger):
                    logger.log_hyperparams(params=params, metrics=metrics)
                    logger.log_graph(self)
                else:
                    logger.log_hyperparams(params=params)

    def training_step(
        self: Self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        losses = self.compute_losses(batch, step_name="train")
        self.log_dict(losses)
        return losses[f"train/{self.hparams.train_loss}"]

    def validation_step(
        self: Self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        losses = self.compute_losses(batch, step_name="val")
        self.log_dict(losses, sync_dist=True)
        metrics = self.update_metrics(batch, step_name="val")
        self.log_dict(metrics)

    def test_step(self: Self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        losses = self.compute_losses(batch, step_name="test")
        self.log_dict(losses, sync_dist=True)
        metrics = self.update_metrics(batch, step_name="test")
        self.log_dict(metrics)

    def predict_step(
        self: Self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.score(batch)

    def on_validation_epoch_end(self: Self) -> None:
        metrics = {
            key.replace("val/", "hp/"): self.trainer.callback_metrics[key]
            for key in self.metrics["val"]
        }
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self: Self) -> torch.optim.Optimizer:
        optimizer_class = (
            torch.optim.SparseAdam if self.model.sparse else torch.optim.AdamW
        )
        return optimizer_class(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self: Self) -> list[Callback]:
        import lightning.pytorch.callbacks as lp_callbacks

        early_stop = lp_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        checkpoint = lp_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"], auto_insert_metric_name=False
        )
        return [early_stop, checkpoint]

    def configure_model(self: Self) -> None:
        if self.model is None:
            self.model = self.get_model()
        if self.metrics is None:
            self.metrics = self.get_metrics(top_k=20)

    def get_model(self: Self) -> torch.nn.Module:
        from functools import partial

        import mf_torch.models as mf_models

        match self.hparams.get("embedder_type"):
            case None:
                embedder_class = mf_models.EmbeddingBag
            case "attention":
                embedder_class = partial(
                    mf_models.AttentionEmbeddingBag,
                    num_heads=self.hparams.num_heads,
                    dropout=self.hparams.dropout,
                )
            case "transformer":
                embedder_class = partial(
                    mf_models.TransformerEmbeddingBag,
                    num_heads=self.hparams.num_heads,
                    dropout=self.hparams.dropout,
                )
            case _:
                msg = f"{self.hparams.embedder_type = }"
                raise NotImplementedError(msg)

        embedder = embedder_class(
            num_embeddings=self.hparams.num_embeddings,
            embedding_dim=self.hparams.embedding_dim,
            max_norm=self.hparams.get("max_norm"),
            norm_type=self.hparams.norm_type,
        )
        model = mf_models.MatrixFactorization(
            embedder=embedder, normalize=self.hparams.normalize
        )
        return torch.jit.script(model)

    @cached_property
    def loss_fns(self: Self) -> torch.nn.ModuleList:
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
            loss_class(hard_negatives_ratio=self.hparams.get("hard_negatives_ratio"))
            for loss_class in loss_classes
        ]
        return torch.nn.ModuleList(loss_fns)

    def get_metrics(self: Self, top_k: int = 20) -> torch.nn.ModuleDict:
        import torchmetrics
        import torchmetrics.retrieval as tm_retrieval

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

    @cached_property
    def example_input_array(self: Self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((1, 1), dtype=torch.int, device=self.device),
            torch.zeros((1, 1), dtype=self.dtype, device=self.device),
        )

    def save_torchscript(self: Self, path: str) -> torch.jit.ScriptModule:
        script_module = torch.jit.script(self.model.eval())
        torch.jit.save(script_module, path)
        return script_module


def cli_main(
    args: ArgsType = None,
    *,
    run: bool = True,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> LightningCLI:
    import lightning.pytorch.callbacks as lp_callbacks
    import mlflow
    from jsonargparse import lazy_instance
    from lightning.pytorch.cli import LightningCLI

    from mf_torch.data.lightning import MatrixFactorizationPipeDataModule
    from mf_torch.params import MLFLOW_DIR, TENSORBOARD_DIR

    mlflow.config.enable_system_metrics_logging()
    tensorboard_logger = {
        "class_path": "TensorBoardLogger",
        "init_args": {
            "save_dir": TENSORBOARD_DIR,
            "name": experiment_name or EXPERIMENT_NAME,
            "version": run_name,
            "log_graph": True,
            "default_hp_metric": False,
        },
    }
    mlflow_logger = {
        "class_path": "MLFlowLogger",
        "init_args": {
            "save_dir": MLFLOW_DIR,
            "experiment_name": experiment_name or EXPERIMENT_NAME,
            "run_name": run_name,
            "log_model": True,
        },
    }
    progress_bar = lazy_instance(lp_callbacks.RichProgressBar)
    trainer_defaults = {
        "precision": "bf16-true",
        "logger": [tensorboard_logger, mlflow_logger],
        "callbacks": [progress_bar],
        "max_epochs": 1,
        "max_time": "00:01:00:00",
    }
    return LightningCLI(
        MatrixFactorizationLitModule,
        MatrixFactorizationPipeDataModule,
        trainer_defaults=trainer_defaults,
        args=args,
        run=run,
    )


if __name__ == "__main__":
    cli_main()
    # cli_main(args={"fit": {"trainer": {"overfit_batches": 1}}})
