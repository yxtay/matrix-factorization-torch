from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import lightning as L
import torch

if TYPE_CHECKING:
    import mlflow
    import torchmetrics

METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}


class LitMatrixFactorization(L.LightningModule):
    def __init__(
        self,
        num_embeddings: int = 2**16 + 1,
        embedding_dim: int = 32,
        train_loss: str = "PairwiseHingeLoss",
        *,
        max_norm: float | None = None,
        sparse: bool = True,
        normalize: bool = True,
        hard_negatives_ratio: float | None = None,
        learning_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.loss_fns = None
        self.metrics = None
        self.example_input_array = None

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_feature_hashes = batch["user_feature_hashes"]
        # shape: (batch_size, num_user_features)
        item_feature_hashes = batch["item_feature_hashes"]
        # shape: (batch_size, num_user_features)
        user_feature_weights = batch["user_feature_weights"]
        # shape: (batch_size, num_item_features)
        item_feature_weights = batch["item_feature_weights"]
        # shape: (batch_size, num_item_features)
        # output shape: (batch_size)
        return self.model(
            user_feature_hashes,
            item_feature_hashes,
            user_feature_weights=user_feature_weights,
            item_feature_weights=item_feature_weights,
        )

    def compute_losses(
        self: LitMatrixFactorization,
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

        user_embed = self.model.embed(user_feature_hashes, user_feature_weights)
        # shape: (batch_size, embed_dim)
        item_embed = self.model.embed(item_feature_hashes, item_feature_weights)
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
            neg_item_embed = self.model.embed(
                neg_item_feature_hashes, neg_item_feature_weights
            )
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
        self: LitMatrixFactorization,
        batch: dict[str, torch.Tensor],
        step_name: str = "train",
    ) -> torchmetrics.MetricCollection:
        user_idx = batch["user_idx"].long()
        label = batch["label"]
        score = self(batch)
        metrics = self.metrics[step_name]
        metrics.update(preds=score, target=label, indexes=user_idx)
        return metrics

    def on_fit_start(self: LitMatrixFactorization) -> None:
        if self.global_rank == 0:
            metrics = {key.replace("val/", "hp/"): 0.0 for key in self.metrics["val"]}
            self.logger.log_hyperparams(params=self.hparams, metrics=metrics)
            self.logger.log_graph(self)

    def training_step(
        self: LitMatrixFactorization, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        losses = self.compute_losses(batch, step_name="train")
        self.log_dict(losses)
        return losses[f"train/{self.hparams.train_loss}"]

    def validation_step(
        self: LitMatrixFactorization, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        losses = self.compute_losses(batch, step_name="val")
        self.log_dict(losses, sync_dist=True)
        metrics = self.update_metrics(batch, step_name="val")
        self.log_dict(metrics)

    def test_step(
        self: LitMatrixFactorization, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        losses = self.compute_losses(batch, step_name="test")
        self.log_dict(losses, sync_dist=True)
        metrics = self.update_metrics(batch, step_name="test")
        self.log_dict(metrics)

    def predict_step(
        self: LitMatrixFactorization, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self(batch)

    def on_validation_epoch_end(self: LitMatrixFactorization) -> None:
        metrics = {
            key.replace("val/", "hp/"): self.trainer.callback_metrics[key]
            for key in self.metrics["val"]
        }
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self: LitMatrixFactorization) -> torch.optim.Optimizer:
        if self.model.sparse:
            optimizer_class = torch.optim.SparseAdam
        else:
            optimizer_class = torch.optim.AdamW
        return optimizer_class(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self: LitMatrixFactorization) -> list[L.Callback]:
        import lightning.pytorch.callbacks as pl_callbacks

        early_stop = pl_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        checkpoint = pl_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"], auto_insert_metric_name=False
        )
        return [early_stop, checkpoint]

    def configure_model(self: LitMatrixFactorization) -> None:
        if self.model is None:
            self.model = self.get_model()
        if self.loss_fns is None:
            self.loss_fns = self.get_loss_fns()
        if self.metrics is None:
            self.metrics = self.get_metrics(top_k=20)
        if self.example_input_array is None:
            self.example_input_array = self.get_example_input_array()

    def get_model(self: LitMatrixFactorization) -> torch.nn.Module:
        from . import models as mf_models

        return mf_models.MatrixFactorization(
            num_embeddings=self.hparams.num_embeddings,
            embedding_dim=self.hparams.embedding_dim,
            max_norm=self.hparams.max_norm,
            sparse=self.hparams.sparse,
            normalize=self.hparams.normalize,
        )

    def get_loss_fns(self: LitMatrixFactorization) -> torch.nn.ModuleList:
        from . import losses as mf_losses

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
            loss_class(hard_negatives_ratio=self.hparams.hard_negatives_ratio)
            for loss_class in loss_classes
        ]
        return torch.nn.ModuleList(loss_fns)

    def get_metrics(
        self: LitMatrixFactorization, top_k: int = 20
    ) -> torch.nn.ModuleDict:
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

    def get_example_input_array(
        self: LitMatrixFactorization,
    ) -> tuple[dict[str, torch.Tensor]]:
        return (
            {
                "user_feature_hashes": torch.zeros(1, 1).int(),
                "item_feature_hashes": torch.zeros(1, 1).int(),
                "user_feature_weights": torch.zeros(1, 1),
                "item_feature_weights": torch.zeros(1, 1),
            },
        )


def mlflow_start_run(experiment_name: str | None = None) -> mlflow.ActiveRun:
    import mlflow

    experiment_name = experiment_name or get_sgt_now().isoformat()
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    return mlflow.start_run(experiment_id=experiment_id)


def get_trainer(
    experiment_name: str | None = None, run_name: str | None = None
) -> L.Trainer:
    import lightning.pytorch.callbacks as pl_callbacks
    import lightning.pytorch.loggers as pl_loggers

    experiment_name = experiment_name or get_sgt_now().isoformat()
    logger = [
        pl_loggers.TensorBoardLogger(
            save_dir="lightning_logs",
            name=experiment_name,
            version=run_name,
            log_graph=True,
            default_hp_metric=False,
        ),
        pl_loggers.MLFlowLogger(
            tracking_uri="mlruns",
            experiment_name=experiment_name,
            run_name=run_name,
            log_model=True,
        ),
    ]
    callbacks = [pl_callbacks.RichProgressBar()]
    return L.Trainer(
        precision="bf16-mixed",
        logger=logger,
        callbacks=callbacks,
        max_epochs=1,
        max_time=datetime.timedelta(hours=1),
        # fast_dev_run=True,
    )


def get_sgt_now() -> datetime.datetime:
    import zoneinfo

    return datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Singapore"))


if __name__ == "__main__":
    import mlflow

    from .data import lightning as mf_data

    experiment_name = get_sgt_now().isoformat()
    train_losses = [
        "PairwiseHingeLoss",
        # "PairwiseLogisticLoss",
        # "InfomationNoiseContrastiveEstimationLoss",
        # "MutualInformationNeuralEstimationLoss",
        # "AlignmentContrastiveLoss",
        # "AlignmentUniformityLoss",
    ]
    for train_loss in train_losses:
        with mlflow_start_run(experiment_name) as run:
            trainer = get_trainer(experiment_name, run.info.run_name)

            with trainer.init_module():
                datamodule = mf_data.Movielens1mPipeDataModule()
                model = LitMatrixFactorization(train_loss=train_loss)

            mlflow.log_params(datamodule.hparams)
            mlflow.log_params(model.hparams)
            trainer.fit(model=model, datamodule=datamodule)
