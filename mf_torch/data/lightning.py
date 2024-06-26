from __future__ import annotations

import abc
import functools
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import torch.utils.data as torch_data
import torch.utils.data.datapipes as torch_datapipes

from mf_torch.data.load import gather_inputs, hash_features, merge_rows, ray_collate_fn
from mf_torch.data.prepare import (
    DATA_DIR,
    MOVIELENS_1M_URL,
    download_unpack_data,
    load_dense_movielens,
)

if TYPE_CHECKING:
    import polars as pl
    import ray.data


class Movielens1mBaseDataModule(L.LightningDataModule, abc.ABC):
    url: str = MOVIELENS_1M_URL
    user_idx: str = "user_idx"
    item_idx: str = "movie_idx"
    label: str = "rating"
    weight: str = "rating"
    user_feature_names: list[str] = [
        "user_id",
        "gender",
        "age",
        "occupation",
        "zipcode",
    ]
    item_feature_names: list[str] = [
        "movie_id",
        "genres",
    ]
    in_columns: list[str] = [
        user_idx,
        item_idx,
        label,
        weight,
        *user_feature_names,
        *item_feature_names,
    ]

    def __init__(
        self: Movielens1mBaseDataModule,
        data_dir: str = DATA_DIR,
        num_hashes: int = 2,
        num_buckets: int = 2**16 + 1,
        batch_size: int = 2**10,
        negatives_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(
        self: Movielens1mBaseDataModule, *, overwrite: bool = False
    ) -> pl.LazyFrame:
        from filelock import FileLock

        with FileLock(f"{self.hparams.data_dir}.lock"):
            download_unpack_data(self.url, self.hparams.data_dir, overwrite=overwrite)
            return load_dense_movielens(self.hparams.data_dir, overwrite=overwrite)

    @classmethod
    @abc.abstractmethod
    def get_dataset(
        cls: type[Movielens1mBaseDataModule], subset: str
    ) -> torch_data.Dataset: ...

    @abc.abstractmethod
    def get_dataloader(
        self: Movielens1mBaseDataModule, dataset: torch_data.Dataset
    ) -> torch_data.DataLoader: ...

    def setup(self: Movielens1mBaseDataModule, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.get_dataset("train")

        if stage in ("fit", "validate"):
            self.val_data = self.get_dataset("val")

        if stage == "test":
            self.test_data = self.get_dataset("test")

        if stage == "predict":
            self.predict_data = self.get_dataset("predict")

    def train_dataloader(
        self: Movielens1mBaseDataModule,
    ) -> torch_data.DataLoader:
        return self.get_dataloader(self.train_data)

    def val_dataloader(self: Movielens1mBaseDataModule) -> torch_data.DataLoader:
        return self.get_dataloader(self.val_data)

    def test_dataloader(self: Movielens1mBaseDataModule) -> torch_data.DataLoader:
        return self.get_dataloader(self.test_data)

    def predict_dataloader(
        self: Movielens1mBaseDataModule,
    ) -> torch_data.DataLoader:
        return self.get_dataloader(self.predict_data)


class Movielens1mPipeDataModule(Movielens1mBaseDataModule):
    def get_movies_dataset(
        self: Movielens1mPipeDataModule,
        cycle_count: int | None = 1,
        *,
        out_prefix: str = "",
        keep_input: bool = True,
    ) -> torch_data.Dataset:
        delta_path = Path(self.hparams.data_dir, "ml-1m", "movies.delta")

        return (
            torch_datapipes.iter.IterableWrapper([delta_path])
            .load_delta_table_as_dict(
                columns=[self.item_idx, *self.item_feature_names],
                batch_size=self.hparams.batch_size,
            )
            .cycle(cycle_count)
            .shuffle(buffer_size=self.hparams.batch_size * 2**4)
            .sharding_filter()
            .map(
                functools.partial(
                    hash_features,
                    idx=self.item_idx,
                    feature_names=self.item_feature_names,
                    num_hashes=self.hparams.num_hashes,
                    num_buckets=self.hparams.num_buckets,
                    out_prefix=out_prefix,
                    keep_input=keep_input,
                )
            )
        )

    def get_dataset(self: Movielens1mPipeDataModule, subset: str) -> torch_data.Dataset:
        import pyarrow.dataset as ds

        assert subset in {"train", "val", "test", "predict"}
        delta_file = {
            "train": "sparse.delta",
            "val": "val.delta",
        }.get(subset, "dense.delta")
        delta_path = Path(self.hparams.data_dir, "ml-1m", delta_file)
        filter_col = f"is_{subset}"

        datapipe = (
            torch_datapipes.iter.IterableWrapper([delta_path])
            .load_delta_table_as_dict(
                columns=list({*self.in_columns}),
                filter_expr=ds.field(filter_col),
                batch_size=self.hparams.batch_size,
            )
            .shuffle(buffer_size=self.hparams.batch_size * 2**4)
            .sharding_filter()
            .map(
                functools.partial(
                    hash_features,
                    idx=self.user_idx,
                    feature_names=self.user_feature_names,
                    num_hashes=self.hparams.num_hashes,
                    num_buckets=self.hparams.num_buckets,
                    out_prefix="user_",
                )
            )
            .map(
                functools.partial(
                    hash_features,
                    idx=self.item_idx,
                    feature_names=self.item_feature_names,
                    num_hashes=self.hparams.num_hashes,
                    num_buckets=self.hparams.num_buckets,
                    out_prefix="item_",
                )
            )
            .map(
                functools.partial(
                    gather_inputs,
                    label=self.label,
                    weight=self.weight,
                )
            )
        )
        if subset == "train" and self.hparams.negatives_ratio > 0:
            datapipe = (
                self.get_movies_dataset(
                    cycle_count=None, out_prefix="neg_item_", keep_input=False
                )
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


class Movielens1mRayDataModule(Movielens1mBaseDataModule):
    def get_dataset(self: Movielens1mRayDataModule, subset: str) -> ray.data.Dataset:
        import deltalake as dl
        import pyarrow.dataset as ds
        import ray.data

        assert subset in {"train", "val", "test", "predict"}
        delta_file = {
            "train": "sparse.delta",
            "val": "val.delta",
        }.get(subset, "dense.delta")
        delta_path = Path(self.hparams.data_dir, "ml-1m", delta_file)
        filter_col = f"is_{subset}"

        parquet_paths = [
            f"{delta_path}/{fragment.path}"
            for fragment in dl.DeltaTable(delta_path)
            .to_pyarrow_dataset()
            .get_fragments(filter=ds.field(filter_col))
        ]
        return (
            ray.data.read_parquet(
                parquet_paths,
                columns=list({*self.in_columns, filter_col}),
                filter=ds.field(filter_col),
            )
            .map(
                hash_features,
                fn_kwargs={
                    "idx": self.user_idx,
                    "feature_names": self.user_feature_names,
                    "num_hashes": self.hparams.num_hashes,
                    "num_buckets": self.hparams.num_buckets,
                    "out_prefix": "user_",
                },
            )
            .map(
                hash_features,
                fn_kwargs={
                    "idx": self.item_idx,
                    "feature_names": self.item_feature_names,
                    "num_hashes": self.hparams.num_hashes,
                    "num_buckets": self.hparams.num_buckets,
                    "out_prefix": "item_",
                },
            )
            .map(
                gather_inputs,
                fn_kwargs={
                    "label": self.label,
                    "weight": self.weight,
                },
            )
        )

    def get_dataloader(
        self: Movielens1mRayDataModule, dataset: ray.data.Dataset
    ) -> ray.data.Dataset:
        return dataset.iter_torch_batches(
            batch_size=self.hparams.batch_size,
            collate_fn=ray_collate_fn,
            local_shuffle_buffer_size=self.hparams.batch_size * 2**4,
        )


if __name__ == "__main__":
    import rich

    for dm_cls in [Movielens1mPipeDataModule, Movielens1mRayDataModule]:
        dm = dm_cls()
        dm.prepare_data().head().collect().glimpse()
        dm.setup("fit")
        for dataloader_fn in [dm.train_dataloader, dm.val_dataloader]:
            batch = next(iter(dataloader_fn()))
            rich.print(batch)
            rich.print({key: value.shape for key, value in batch.items()})
