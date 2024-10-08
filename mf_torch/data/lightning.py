from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.utils.data as torch_data
import torch.utils.data.datapipes as torch_datapipes
from lightning import LightningDataModule

from mf_torch.params import (
    DATA_DIR,
    ITEM_FEATURE_NAMES,
    ITEM_IDX,
    MOVIELENS_1M_URL,
    NUM_EMBEDDINGS,
    NUM_HASHES,
    USER_FEATURE_NAMES,
    USER_IDX,
)

if TYPE_CHECKING:
    from typing import Self

    import polars as pl


LABEL = "rating"
WEIGHT = "rating"


class MatrixFactorizationDataModule(LightningDataModule):
    label: str = LABEL
    weight: str = WEIGHT
    user_idx: str = USER_IDX
    user_features: list[str] = USER_FEATURE_NAMES
    item_idx: str = ITEM_IDX
    item_features: list[str] = ITEM_FEATURE_NAMES

    def __init__(
        self: Self,
        data_dir: str = DATA_DIR,
        num_hashes: int = NUM_HASHES,
        num_embeddings: int = NUM_EMBEDDINGS,
        negatives_ratio: int = 1,
        batch_size: int = 2**10,
        num_workers: int | None = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self: Self, *, overwrite: bool = False) -> pl.LazyFrame:
        from filelock import FileLock

        from mf_torch.data.prepare import download_unpack_data, load_dense_movielens

        with FileLock(f"{self.hparams.data_dir}.lock"):
            download_unpack_data(
                MOVIELENS_1M_URL, self.hparams.data_dir, overwrite=overwrite
            )
            return load_dense_movielens(self.hparams.data_dir, overwrite=overwrite)

    def get_raw_items_data(self: Self) -> torch_data.IterDataPipe:
        delta_path = Path(self.hparams.data_dir, "ml-1m", "movies.delta")
        return torch_datapipes.iter.IterableWrapper(
            [delta_path]
        ).load_delta_table_as_dict(batch_size=self.hparams.batch_size)

    def get_items_dataset(
        self: Self, *, cycle_count: int | None = 1, prefix: str = "item_"
    ) -> torch_data.IterDataPipe:
        from mf_torch.data.load import process_features

        return (
            self.get_raw_items_data()
            .cycle(cycle_count)
            .shuffle(buffer_size=self.hparams.batch_size * 2**4)
            .sharding_filter()
            .map(
                partial(
                    process_features,
                    idx=self.item_idx,
                    feature_names=self.item_features,
                    prefix=prefix,
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
        )

    def get_raw_data(self: Self, subset: str) -> torch_data.IterDataPipe:
        import pyarrow.dataset as ds

        from mf_torch.data.load import score_interactions

        assert subset in {"train", "val", "test", "predict"}
        filter_col = f"is_{subset}"

        delta_file = {
            "train": "sparse.delta",
            "val": "val.delta",
        }.get(subset, "dense.delta")
        delta_path = Path(self.hparams.data_dir, "ml-1m", delta_file)

        columns = {
            self.label,
            self.weight,
            self.item_idx,
            *self.item_features,
            self.user_idx,
            *self.user_features,
        }
        return (
            torch_datapipes.iter.IterableWrapper([delta_path])
            .load_delta_table_as_dict(
                columns=list(columns),
                filter_expr=ds.field(filter_col),
                batch_size=self.hparams.batch_size,
            )
            .map(partial(score_interactions, label=self.label, weight=self.weight))
        )

    def get_dataset(self: Self, subset: str) -> torch_data.IterDataPipe:
        from mf_torch.data.load import merge_rows, process_features, select_fields

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
            self.get_raw_data(subset)
            .shuffle(buffer_size=self.hparams.batch_size * 2**4)
            .sharding_filter()
            .map(
                partial(
                    process_features,
                    idx=self.item_idx,
                    feature_names=self.item_features,
                    prefix="item_",
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
            .map(
                partial(
                    process_features,
                    idx=self.user_idx,
                    feature_names=self.user_features,
                    prefix="user_",
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
            .map(partial(select_fields, fields=fields))
        )
        if subset == "train" and self.hparams.negatives_ratio > 0:
            fields = [
                "neg_item_idx",
                "neg_item_feature_hashes",
                "neg_item_feature_weights",
            ]
            items_datapipe = (
                self.get_items_dataset(cycle_count=None, prefix="neg_item_")
                .map(partial(select_fields, fields=fields))
                .batch(self.hparams.negatives_ratio)
                .collate()
            )
            datapipe = datapipe.zip(items_datapipe).map(merge_rows)

        return datapipe

    def get_dataloader(
        self: Self, dataset: torch_data.Dataset
    ) -> torch_data.DataLoader:
        import os

        num_workers = self.hparams.get("num_workers")
        if num_workers is None:
            num_workers = os.cpu_count() - 1
        mp_ctx = "spawn" if num_workers > 0 else None

        return torch_data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            multiprocessing_context=mp_ctx,
            persistent_workers=num_workers > 0,
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

    dm = MatrixFactorizationDataModule(num_workers=1)
    dm.prepare_data().head().collect().glimpse()
    dm.setup("fit")

    dataloaders = [
        dm.get_items_dataset(),
        dm.train_data,
        dm.val_data,
        dm.train_dataloader(),
        dm.val_dataloader(),
    ]
    for dataloader in dataloaders:
        batch = next(iter(dataloader))
        rich.print(batch)
        shapes = {
            key: value.shape
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        rich.print(shapes)
