import lightning as L
import torch
import torchmetrics
from lightning.pytorch import callbacks as pl_callbacks
from torchmetrics import retrieval as tm_retrieval

from . import losses as mf_losses
from . import models as mf_models


class LitMatrixFactorization(L.LightningModule):
    def __init__(
        self,
        num_embeddings: int = 2**16 + 1,
        embedding_dim: int = 32,
        *,
        train_loss: str = "PairwiseLogisticLoss",
        max_norm: float = None,
        sparse: bool = True,
        learning_rate: float = 1.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = mf_models.MatrixFactorization(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
            sparse=sparse,
            normalize=normalize,
        )
        self.example_input_array = self.get_example_input_array()
        self.loss_fns = self.get_loss_fns()
        self.metrics = self.get_metrics(top_k=20)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_features = batch["user_features"]
        item_features = batch["item_features"]
        user_feature_weights = batch.get("user_feature_weights")
        item_feature_weights = batch.get("item_feature_weights")
        return self.model(
            user_features,
            item_features,
            user_feature_weights=user_feature_weights,
            item_feature_weights=item_feature_weights,
        )

    def compute_losses(self, batch, step_name="train"):
        label = batch["label"]
        sample_weight = batch["weight"]
        user_features = batch["user_features"]
        item_features = batch["item_features"]
        user_feature_weights = batch.get("user_feature_weights")
        item_feature_weights = batch.get("item_feature_weights")
        user_idx = batch["user_idx"]
        item_idx = batch["item_idx"]

        user_embed = self.model.embed(user_features, user_feature_weights)
        item_embed = self.model.embed(item_features, item_feature_weights)

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

    def update_metrics(self, batch, step_name):
        user_idx = batch["user_idx"].long()
        label = batch["label"]
        score = self(batch)
        metrics = self.metrics[step_name]
        metrics.update(preds=score, target=label, indexes=user_idx)
        return metrics

    def on_fit_start(self) -> None:
        self.logger.log_graph(self)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        losses = self.compute_losses(batch, step_name="train")
        self.log_dict(losses)
        return losses[f"train/{self.hparams.train_loss}"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        losses = self.compute_losses(batch, step_name="val")
        self.log_dict(losses)
        metrics = self.update_metrics(batch, step_name="val")
        self.log_dict(metrics)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        losses = self.compute_losses(batch, step_name="test")
        self.log_dict(losses)
        metrics = self.update_metrics(batch, step_name="test")
        self.log_dict(metrics)

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self(batch)

    def on_validation_end(self):
        if self.current_epoch == 0 and self.global_step > 0:
            metrics = {
                key.replace("val/", "metric/"): value
                for key, value in self.trainer.callback_metrics.items()
                if key.startswith("val/")
            }
            self.logger.log_hyperparams(params=self.hparams, metrics=metrics)

    # def on_fit_end(self) -> None:
    #     if self.trainer.is_global_zero:
    #         tb_logger = self.logger.experiment
    #         tb_logger.add_embedding(
    #             self.model.embedding.weight, global_step=self.global_step
    #         )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.model.sparse:
            optimizer_class = torch.optim.SparseAdam
        else:
            optimizer_class = torch.optim.AdamW
        return optimizer_class(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self) -> list[L.Callback]:
        monitor = "val/RetrievalNormalizedDCG"
        early_stop = pl_callbacks.EarlyStopping(monitor=monitor, mode="max")
        checkpoint = pl_callbacks.ModelCheckpoint(
            monitor=monitor, mode="max", auto_insert_metric_name=False
        )
        return [early_stop, checkpoint]

    def get_loss_fns(self) -> torch.nn.Module:
        loss_fns = [
            mf_losses.AlignmentLoss(),
            mf_losses.AlignmentUniformityLoss(),
            mf_losses.ItemUniformityLoss(),
            mf_losses.UserUniformityLoss(),
            mf_losses.ContrastiveLoss(),
            mf_losses.AlignmentContrastiveLoss(),
            mf_losses.MutualInformationNeuralEstimatorLoss(),
            mf_losses.PairwiseExponentialLoss(),
            mf_losses.PairwiseHingeLoss(),
            mf_losses.PairwiseLogisticLoss(),
        ]
        return torch.nn.ModuleList(loss_fns)

    def get_metrics(self, top_k: int = 20) -> torch.nn.Module:
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

    def get_example_input_array(self) -> tuple[dict[str, torch.Tensor]]:
        example_input_array = (
            {
                "user_features": torch.zeros(1, 1).int(),
                "item_features": torch.zeros(1, 1).int(),
                "user_feature_weights": torch.zeros(1, 1),
                "item_feature_weights": torch.zeros(1, 1),
            },
        )
        return example_input_array


if __name__ == "__main__":
    import datetime
    import itertools

    from lightning.pytorch import loggers as pl_loggers

    from .data import load as dm

    train_losses = [
        "PairwiseLogisticLoss",
        "MutualInformationNeuralEstimatorLoss",
        "AlignmentUniformityLoss",
        "AlignmentContrastiveLoss",
        "PairwiseHingeLoss",
    ]

    for normalize, train_loss in itertools.product([True, False], train_losses):
        tb_logger = pl_loggers.TensorBoardLogger(
            ".", log_graph=True, default_hp_metric=False
        )
        callbacks = [pl_callbacks.RichProgressBar()]
        trainer = L.Trainer(
            precision="bf16-mixed",
            logger=tb_logger,
            callbacks=callbacks,
            max_epochs=1,
            max_time=datetime.timedelta(hours=1),
            # fast_dev_run=True,
        )
        model = LitMatrixFactorization(train_loss=train_loss, normalize=normalize)
        datamodule = dm.Movielens1mPipeDataModule(batch_size=2**10)
        trainer.fit(model=model, datamodule=datamodule)
