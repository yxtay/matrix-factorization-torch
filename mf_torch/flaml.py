from __future__ import annotations

from typing import TYPE_CHECKING

from mf_torch.params import METRIC

if TYPE_CHECKING:
    import flaml.tune.tune


def get_lightning_args(
    config: dict[str, bool | float | int | str],
) -> dict[str, bool | float | int | str]:
    num_embeddings = 2 ** config["log_num_embeddings"] + 1
    embedding_dim = 2 ** config["log_embedding_dim"]
    # num_heads = 2 ** config["log_num_heads"]

    hard_negatives_ratio = (
        config["hard_negatives_ratio"] if config["use_hard_negatives"] else None
    )

    model_args = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "train_loss": config["train_loss"],
        # "embedder_type": config["embedder_type"],
        # "num_heads": num_heads,
        # "dropout": config["dropout"],
        "hard_negatives_ratio": hard_negatives_ratio,
        "learning_rate": config["learning_rate"],
    }
    data_args = {
        "num_hashes": config["num_hashes"],
        "num_embeddings": num_embeddings,
    }
    return {"model": model_args, "data": data_args}


def evaluation_function(
    config: dict[str, bool | float | int | str],
) -> dict[str, float]:
    import numpy as np

    from mf_torch.lightning import cli_main

    config = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in config.items()
    }

    trainer_args = {
        "limit_train_batches": config["limit_train_batches"],
        "num_sanity_val_steps": 0,
    }
    args = {"fit": {"trainer": trainer_args, **get_lightning_args(config)}}
    cli = cli_main(args)
    return {
        key: cli.trainer.callback_metrics[key].item()
        for key in cli.model.metrics["val"]
    }


def flaml_tune() -> flaml.tune.tune.ExperimentAnalysis:
    import flaml.tune

    train_losses = [
        "PairwiseHingeLoss",
        "PairwiseLogisticLoss",
        "AlignmentContrastiveLoss",
        "MutualInformationNeuralEstimationLoss",
    ]
    config = {
        "num_hashes": flaml.tune.randint(1, 5),
        "log_num_embeddings": flaml.tune.randint(12, 21),
        "log_embedding_dim": flaml.tune.randint(5, 9),
        # "embedder_type": flaml.tune.choice([None, "attention", "transformer"]),
        # "log_num_heads": flaml.tune.randint(0, 3),
        # "dropout": flaml.tune.quniform(0.0, 0.5, 0.01),
        "train_loss": flaml.tune.choice(train_losses),
        "use_hard_negatives": flaml.tune.choice([True, False]),
        "hard_negatives_ratio": flaml.tune.quniform(0.5, 2.0, 0.01),
        "learning_rate": flaml.tune.qloguniform(0.001, 0.1, 0.001),
    }
    low_cost_partial_config = {}
    point_to_evaluate = {
        "num_hashes": 2,
        "log_num_embeddings": 16,
        "log_embedding_dim": 5,
        # "embedder_type": None,
        # "log_num_heads": 0,
        # "dropout": 0.0,
        "train_loss": "PairwiseHingeLoss",
        "use_hard_negatives": False,
        "hard_negatives_ratio": 1.0,
        "learning_rate": 0.1,
    }
    return flaml.tune.run(
        evaluation_function,
        metric=METRIC["name"],
        mode=METRIC["mode"],
        config=config,
        low_cost_partial_config=low_cost_partial_config,
        points_to_evaluate=[point_to_evaluate],
        time_budget_s=60 * 60 * 1,
        num_samples=-1,
        resource_attr="limit_train_batches",
        min_resource=0.25,
        max_resource=1.0,
        reduction_factor=2,
    )


if __name__ == "__main__":
    import rich

    analysis = flaml_tune()
    rich.print(analysis.best_result)
