from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ray.train
import ray.tune

if TYPE_CHECKING:
    import lightning as L
    import ray.train.torch as ray_torch


def prepare_trainer(config: dict[str, bool | float | int | str]) -> L.Trainer:
    import lightning as L
    import lightning.pytorch.loggers as lp_loggers
    import ray.train.lightning as ray_lightning

    experiment_name = ray.train.get_context().get_experiment_name()
    trial_name = ray.train.get_context().get_trial_name()
    logger = [
        lp_loggers.TensorBoardLogger(
            save_dir=config["tensorboard_save_dir"],
            name=experiment_name or "lightning_logs",
            version=trial_name,
            log_graph=True,
            default_hp_metric=False,
        ),
        lp_loggers.MLFlowLogger(
            tracking_uri=str(config["mlflow_tracking_uri"]),
            experiment_name=experiment_name,
            run_name=trial_name,
            log_model=True,
        ),
    ]
    trainer = L.Trainer(
        precision=config["precision"],
        max_epochs=config["max_epochs"],
        max_time=config["max_time"],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=ray_lightning.RayDDPStrategy(),
        logger=logger,
        callbacks=[ray_lightning.RayTrainReportCallback()],
        plugins=[ray_lightning.RayLightningEnvironment()],
    )
    return ray_lightning.prepare_trainer(trainer)


def train_loop_per_worker(config: dict[str, bool | float | int | str]) -> None:
    import ray.train.lightning as ray_lightning

    from mf_torch.data.lightning import Movielens1mPipeDataModule
    from mf_torch.lightning import LitMatrixFactorization

    batch_size = 2 ** (config["batch_size_exp"] - config["negatives_ratio_exp"])
    negatives_ratio = 2 ** config["negatives_ratio_exp"] - 1

    num_embeddings = 2 ** config["num_embeddings_exp"] + 1
    max_norm = 2.0 ** config["max_norm_exp"] if config["use_max_norm"] else None
    num_heads = 2 ** config["num_heads_exp"]

    hard_negatives_ratio = (
        config["hard_negatives_ratio"] if config["use_hard_negatives"] else None
    )

    trainer = prepare_trainer(config)
    with trainer.init_module():
        datamodule = Movielens1mPipeDataModule(
            data_dir=config["data_dir"],
            batch_size=batch_size,
            num_hashes=config["num_hashes"],
            num_embeddings=num_embeddings,
            negatives_ratio=negatives_ratio,
        )
        model = LitMatrixFactorization(
            num_embeddings=num_embeddings,
            embedding_dim=2 ** config["embedding_dim_exp"],
            train_loss=config["train_loss"],
            max_norm=max_norm,
            sparse=config["sparse"],
            embedder_type=config["embedder_type"],
            num_heads=num_heads,
            dropout=config["dropout"],
            normalize=config["normalize"],
            hard_negatives_ratio=hard_negatives_ratio,
            learning_rate=config["learning_rate"],
        )

    ckpt_path = None
    if checkpoint := ray.train.get_checkpoint():
        checkpoint_name = ray_lightning.RayTrainReportCallback.CHECKPOINT_NAME
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = Path(ckpt_dir, checkpoint_name)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def get_run_config() -> ray.train.RunConfig:
    import ray.tune.stopper as ray_stopper

    from mf_torch.lightning import METRIC

    checkpoint_config = ray.train.CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute=METRIC["name"],
        checkpoint_score_order=METRIC["mode"],
    )
    stopper = ray_stopper.ExperimentPlateauStopper(
        metric=METRIC["name"], mode=METRIC["mode"]
    )
    return ray.train.RunConfig(
        storage_path=Path("ray_results").absolute(),
        checkpoint_config=checkpoint_config,
        stop=stopper,
    )


def get_ray_trainer() -> ray_torch.TorchTrainer:
    import os

    import ray.train.torch as ray_torch

    from mf_torch.params import DATA_DIR, MLFLOW_DIR, TENSORBOARD_DIR

    train_loop_config = {
        # tracking
        "tensorboard_save_dir": Path(TENSORBOARD_DIR).absolute(),
        "mlflow_tracking_uri": Path(MLFLOW_DIR).absolute(),
        # trainer
        "precision": "bf16-true",
        "max_epochs": 1,
        "max_time": "00:01:00:00",
        # datamodule
        "data_dir": Path(DATA_DIR).absolute(),
        "batch_size_exp": 11,
        "num_hashes": 2,
        "negatives_ratio_exp": 1,
        # model
        "num_embeddings_exp": 16,
        "embedding_dim_exp": 5,
        "use_max_norm": False,
        "max_norm_exp": 0.0,
        "sparse": False,
        "embedder_type": None,
        "num_heads_exp": 0,
        "dropout": 0.0,
        "normalize": True,
        # lightning module
        "train_loss": "PairwiseHingeLoss",
        "use_hard_negatives": False,
        "hard_negatives_ratio": 1.0,
        "learning_rate": 0.1,
    }
    scaling_config = ray.train.ScalingConfig(
        num_workers=1,
        resources_per_worker={"CPU": os.cpu_count() - 1},
    )
    return ray_torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=get_run_config(),
    )


def get_tuner() -> ray.tune.Tuner:
    import flaml

    from mf_torch.lightning import METRIC

    search_space = {
        # "num_hashes": ray.tune.randint(1, 5),
        # "negatives_ratio_exp": ray.tune.randint(1, 4),
        # "num_embeddings_exp": ray.tune.randint(10, 17),
        # "embedding_dim_exp": ray.tune.randint(2, 7),
        # "use_max_norm": ray.tune.choice([False, True]),
        # "max_norm_exp": ray.tune.randint(0, 6),
        "embedder_type": ray.tune.choice(["attention", "transformer"]),
        "num_heads_exp": ray.tune.randint(0, 3),
        "dropout": ray.tune.quniform(0.0, 0.5, 0.01),
        # "normalize": ray.tune.choice([True, False]),
        # "train_loss": ray.tune.choice(train_losses),
        # "use_hard_negatives": ray.tune.choice([True, False]),
        # "hard_negatives_ratio": ray.tune.quniform(0.5, 2.0, 0.01),
        "learning_rate": ray.tune.qloguniform(0.001, 0.1, 0.001),
        # "precision": ray.tune.choice(["bf16-true", "bf16-mixed"]),
    }
    low_cost_partial_config = {
        # "num_hashes": 1,
        # "negatives_ratio_exp": 1,
        # "num_embeddings_exp": 10,
        # "embedding_dim_exp": 2,
        # "use_max_norm": False,
        # "max_norm_exp": 0,
        # "embedder_type": None,
        "num_heads_exp": 0,
        "dropout": 0.0,
        # "normalize": True,
        # "train_loss": "PairwiseHingeLoss",
        # "use_hard_negatives": True,
        # "hard_negatives_ratio": 1.0,
        # "learning_rate": 0.1,
        # "precision": "bf16-true",
    }
    point_to_evaluate = {
        # "num_hashes": 2,
        # "negatives_ratio_exp": 1,
        # "num_embeddings_exp": 16,
        # "embedding_dim_exp": 5,
        # "train_loss": "PairwiseHingeLoss",
        # "use_max_norm": False,
        # "max_norm_exp": 0,
        # "embedder_type": None,
        "num_heads_exp": 0,
        "dropout": 0.0,
        # "normalize": True,
        # "use_hard_negatives": True,
        # "hard_negatives_ratio": 1.0,
        # "learning_rate": 0.1,
        # "precision": "bf16-true",
    }
    search_alg = flaml.BlendSearch(
        low_cost_partial_config={"train_loop_config": low_cost_partial_config},
        points_to_evaluate=[{"train_loop_config": point_to_evaluate}],
    )
    tune_config = ray.tune.TuneConfig(
        metric=METRIC["name"],
        mode=METRIC["mode"],
        search_alg=search_alg,
        num_samples=-1,
        time_budget_s=60 * 60 * 1,
        max_concurrent_trials=1,
    )
    return ray.tune.Tuner(
        get_ray_trainer(),
        param_space={"train_loop_config": search_space},
        tune_config=tune_config,
        run_config=get_run_config(),
    )


def main() -> None:
    import rich

    tuner = get_tuner()
    results = tuner.fit()
    rich.print(results.experiment_path)
    best_results = results.get_best_result()
    rich.print(best_results.path)
    rich.print(best_results.metrics)


if __name__ == "__main__":
    main()
