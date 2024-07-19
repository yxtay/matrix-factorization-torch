from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import torch.utils.data as torch_data
import torch.utils.data.datapipes as torch_datapipes

from mf_torch.data.load import (
    merge_rows,
    process_features,
    score_interactions,
    select_fields,
)
from mf_torch.data.prepare import download_unpack_data, load_dense_movielens
from mf_torch.params import (
    DATA_DIR,
    ITEM_FEATURE_NAMES,
    ITEM_IDX,
    MOVIELENS_1M_URL,
    USER_FEATURE_NAMES,
    USER_IDX,
)

if TYPE_CHECKING:
    from typing import Self

    import polars as pl


LABEL = "rating"
WEIGHT = "rating"


class Movielens1mPipeDataModule(L.LightningDataModule):
    def __init__(
        self: Self,
        data_dir: str = DATA_DIR,
        num_hashes: int = 2,
        num_embeddings: int = 2**16 + 1,
        batch_size: int = 2**10,
        negatives_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self: Self, *, overwrite: bool = False) -> pl.LazyFrame:
        from filelock import FileLock

        with FileLock(f"{self.hparams.data_dir}.lock"):
            download_unpack_data(
                MOVIELENS_1M_URL, self.hparams.data_dir, overwrite=overwrite
            )
            return load_dense_movielens(self.hparams.data_dir, overwrite=overwrite)

    def get_movies_dataset(
        self: Self, *, cycle_count: int | None = 1, prefix: str = "item_"
    ) -> torch_data.Dataset:
        delta_path = Path(self.hparams.data_dir, "ml-1m", "movies.delta")
        columns = {ITEM_IDX, *ITEM_FEATURE_NAMES}
        return (
            torch_datapipes.iter.IterableWrapper([delta_path])
            .load_delta_table_as_dict(
                columns=list(columns),
                batch_size=self.hparams.batch_size,
            )
            .cycle(cycle_count)
            .shuffle(buffer_size=self.hparams.batch_size * 2**4)
            .sharding_filter()
            .map(
                partial(
                    process_features,
                    idx=ITEM_IDX,
                    feature_names=ITEM_FEATURE_NAMES,
                    prefix=prefix,
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
        )

    def get_dataset(self: Self, subset: str) -> torch_data.Dataset:
        import pyarrow.dataset as ds

        assert subset in {"train", "val", "test", "predict"}
        delta_file = {
            "train": "sparse.delta",
            "val": "val.delta",
        }.get(subset, "dense.delta")
        delta_path = Path(self.hparams.data_dir, "ml-1m", delta_file)
        filter_col = f"is_{subset}"

        columns = {
            LABEL,
            WEIGHT,
            USER_IDX,
            ITEM_IDX,
            *USER_FEATURE_NAMES,
            *ITEM_FEATURE_NAMES,
        }
        fields = [
            "label",
            "weight",
            "user_idx",
            "user_feature_hashes",
            "user_feature_weights",
            "item_idx",
            "item_feature_hashes",
            "item_feature_weights",
        ]
        datapipe = (
            torch_datapipes.iter.IterableWrapper([delta_path])
            .load_delta_table_as_dict(
                columns=list(columns),
                filter_expr=ds.field(filter_col),
                batch_size=self.hparams.batch_size,
            )
            .shuffle(buffer_size=self.hparams.batch_size * 2**4)
            .sharding_filter()
            .map(
                partial(
                    process_features,
                    idx=ITEM_IDX,
                    feature_names=ITEM_FEATURE_NAMES,
                    prefix="item_",
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
            .map(
                partial(
                    process_features,
                    idx=USER_IDX,
                    feature_names=USER_FEATURE_NAMES,
                    prefix="user_",
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
            .map(partial(score_interactions, label=LABEL, weight=WEIGHT))
            .map(partial(select_fields, fields=fields))
        )
        if subset == "train" and self.hparams.negatives_ratio > 0:
            fields = [
                "neg_item_idx",
                "neg_item_feature_hashes",
                "neg_item_feature_weights",
            ]
            datapipe = (
                self.get_movies_dataset(cycle_count=None, prefix="neg_item_")
                .map(partial(select_fields, fields=fields))
                .batch(self.hparams.negatives_ratio)
                .collate()
                .zip(datapipe)
                .map(merge_rows)
            )

        return datapipe

    def get_dataloader(
        self: Movielens1mPipeDataModule, dataset: torch_data.Dataset
    ) -> torch_data.DataLoader:
        import os

        return torch_data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() - 1,
            persistent_workers=True,
        )

    def setup(self: Self, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.get_dataset("train")

        if stage in ("fit", "validate"):
            self.val_data = self.get_dataset("val")

        if stage == "test":
            self.test_data = self.get_dataset("test")

        if stage == "predict":
            self.predict_data = self.get_dataset("predict")

    def train_dataloader(self: Self) -> torch_data.DataLoader:
        return self.get_dataloader(self.train_data)

    def val_dataloader(self: Self) -> torch_data.DataLoader:
        return self.get_dataloader(self.val_data)

    def test_dataloader(self: Self) -> torch_data.DataLoader:
        return self.get_dataloader(self.test_data)

    def predict_dataloader(self: Self) -> torch_data.DataLoader:
        return self.get_dataloader(self.predict_data)


if __name__ == "__main__":
    import rich

    dm = Movielens1mPipeDataModule()
    dm.prepare_data().head().collect().glimpse()
    dm.setup("fit")
    for dataloader_fn in [dm.train_dataloader, dm.val_dataloader]:
        batch = next(iter(dataloader_fn()))
        rich.print(batch)
        rich.print({key: value.shape for key, value in batch.items()})
