import abc
import functools
import os
from collections.abc import Iterable
from pathlib import Path

import lightning as L
import mmh3
import numpy as np
import pyarrow.dataset as ds
import ray
import torch
import torch.utils.data._utils.collate as torch_collate
import torch.utils.data.datapipes as dp

from .prepare import (
    DATA_DIR,
    MOVIELENS_1M_URL,
    download_unpack_data,
    load_dense_movielens,
)


def hash_features(
    row: dict[str, int | float | str | list[int, str]],
    *,
    feature_names: list[str],
    num_hashes: int = 2,
    num_buckets: int = 2**16 + 1,
    out_prefix: str = "",
) -> np.ndarray:
    feature_values = []
    feature_weights = []
    # categorical features
    cat_features = [
        f"{key}:{row[key]}" for key in feature_names if isinstance(row[key], (int, str))
    ]
    if len(cat_features) > 0:
        feature_values.extend(cat_features)
        feature_weights.append(np.ones(len(cat_features)))

    # float features
    float_features = [key for key in feature_names if isinstance(row[key], float)]
    if len(float_features) > 0:
        feature_values.extend(float_features)
        feature_weights.append(np.array([row[key] for key in float_features]))

    # multi categorical features
    for key in feature_names:
        if isinstance(row[key], Iterable) and not isinstance(row[key], str):
            iter_feature_values = [
                f"{key}:{value}" for value in row[key] if isinstance(value, (int, str))
            ]
            num_values = len(iter_feature_values)
            if num_values == 0:
                continue
            iter_feature_weights = np.ones(num_values) / num_values

            feature_values.extend(iter_feature_values)
            feature_weights.append(iter_feature_weights)

    feature_hashes = [
        mmh3.hash(values, seed)
        for seed in range(num_hashes)
        for values in feature_values
    ]
    feature_hashes = np.array(feature_hashes) % (num_buckets - 1) + 1
    feature_weights = np.tile(np.concatenate(feature_weights), num_hashes)

    row[f"{out_prefix}features"] = feature_hashes.astype("int32")
    row[f"{out_prefix}feature_weights"] = feature_weights.astype("float32")
    return row


def gather_inputs(
    row: dict[str, int | float | list[int | float] | None],
    *,
    user_idx: str = "user_idx",
    item_idx: str = "item_idx",
    label: str = "label",
    weight: str = "weight",
) -> dict[str, int | float | list[int | float]]:
    label_value = row[label] or 0
    inputs = {
        "user_idx": row[user_idx] or 0,
        "item_idx": row[item_idx] or 0,
        "label": bool(label_value > 0) - bool(label_value < 0),
        "weight": row[weight] or 0,
        "user_features": row["user_features"],
        "user_feature_weights": row["user_feature_weights"],
        "item_features": row["item_features"],
        "item_feature_weights": row["item_feature_weights"],
    }
    return inputs


def collate_fn(
    batch: np.ndarray | dict[str, np.ndarray],
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return {
            col_name: collate_fn(col_batch) for col_name, col_batch in batch.items()
        }

    # batch is np.ndarray
    if batch.dtype.type is np.object_ and isinstance(batch[0], np.ndarray):
        # batch is ndarray of ndarray if shape is different
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(arr) for arr in batch], batch_first=True
        )
        return padded

    return torch.as_tensor(batch)


def collate_tensor_fn(
    batch: Iterable[torch.Tensor], *, collate_fn_map: dict | None = None
) -> torch.Tensor:
    it = iter(batch)
    elem_size = next(it).size()
    if not all(elem.size() == elem_size for elem in it):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return torch_collate.collate_tensor_fn(batch, collate_fn_map=collate_fn_map)


torch_collate.default_collate_fn_map[torch.Tensor] = collate_tensor_fn


class ParquetIterDataPipe(dp.datapipe.IterDataPipe):
    def __init__(
        self,
        sources: Iterable[str | Path],
        *,
        columns: Iterable[str] | None = None,
        filter_col: str = "",
        batch_size: int = 2**10,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.columns = set(columns) or None
        self.filter_col = filter_col
        self.batch_size = batch_size

        # if both columns and filter_col are specified
        # add filer_col into columns
        if columns and filter_col:
            self.columns.add(filter_col)

    def __len__(self):
        filter_expr = ds.field(self.filter_col) if self.filter_col else None
        return ds.dataset(self.sources).count_rows(filter=filter_expr)

    def __iter__(self):
        dataset = ds.dataset(self.sources)
        if self.filter_col:
            dataset = dataset.filter(ds.field(self.filter_col))
        for batch in dataset.to_batches(
            columns=list(self.columns), batch_size=self.batch_size
        ):
            yield from batch.to_pylist()


class Movielens1mBaseDataModule(L.LightningDataModule, abc.ABC):
    url: str = MOVIELENS_1M_URL
    user_idx: str = "user_idx"
    item_idx: str = "movie_idx"
    label: str = "rating"
    weight: str = "rating"
    user_features: list[str] = ["user_id", "gender", "age", "occupation", "zipcode"]
    item_features: list[str] = ["movie_id", "genres"]
    in_columns: list[str] = [
        user_idx,
        item_idx,
        label,
        weight,
        *user_features,
        *item_features,
    ]

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        num_hashes: int = 2,
        num_buckets: int = 2**16 + 1,
        batch_size: int = 2**10,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.batch_size = batch_size

    def prepare_data(self, *, overwrite: bool = False):
        download_unpack_data(self.url, self.data_dir, overwrite=overwrite)
        return load_dense_movielens(self.data_dir, overwrite=overwrite)

    @abc.abstractclassmethod
    def get_dataset(self, subset: str) -> torch.utils.data.Dataset:
        ...

    @abc.abstractmethod
    def get_dataloader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader:
        ...

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.get_dataset("train")

        if stage in ("fit", "validate"):
            self.val_data = self.get_dataset("val")

        if stage == "test":
            self.test_data = self.get_dataset("test")

        if stage == "predict":
            self.predict_data = self.get_dataset("predict")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader(self.train_data)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader(self.val_data)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader(self.test_data)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader(self.predict_data)


class Movielens1mPipeDataModule(Movielens1mBaseDataModule):
    def get_dataset(self, subset: str) -> torch.utils.data.Dataset:
        assert subset in {"train", "val", "test", "predict"}
        parquet_file = {
            "train": "sparse.parquet",
            "val": "val.parquet",
        }.get(subset, "dense.parquet")
        parquet_path = Path(self.data_dir, "ml-1m", parquet_file)
        filter_col = f"is_{subset}"

        ds = (
            ParquetIterDataPipe(
                [parquet_path],
                columns=self.in_columns,
                filter_col=filter_col,
                batch_size=self.batch_size,
            )
            .shuffle(buffer_size=self.batch_size * 2**4)
            .sharding_filter()
            .map(
                functools.partial(
                    hash_features,
                    feature_names=self.user_features,
                    num_hashes=self.num_hashes,
                    num_buckets=self.num_buckets,
                    out_prefix="user_",
                )
            )
            .map(
                functools.partial(
                    hash_features,
                    feature_names=self.item_features,
                    num_hashes=self.num_hashes,
                    num_buckets=self.num_buckets,
                    out_prefix="item_",
                )
            )
            .map(
                functools.partial(
                    gather_inputs,
                    user_idx=self.user_idx,
                    item_idx=self.item_idx,
                    label=self.label,
                    weight=self.weight,
                )
            )
        )
        return ds

    def get_dataloader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() - 1,
            persistent_workers=True,
        )


class Movielens1mRayDataModule(Movielens1mBaseDataModule):
    def get_dataset(
        self, subset: str, *, materialize: bool = False
    ) -> ray.data.Dataset:
        assert subset in {"train", "val", "test", "predict"}
        parquet_file = {
            "train": "sparse.parquet",
            "val": "val.parquet",
        }.get(subset, "dense.parquet")
        parquet_path = Path(self.data_dir, "ml-1m", parquet_file)
        filter_col = f"is_{subset}"

        ds = (
            ray.data.read_parquet(
                parquet_path,
                columns=list({*self.in_columns}),
                filter=ds.field(filter_col),
            )
            .map(
                hash_features,
                fn_kwargs={
                    "feature_names": self.user_features,
                    "num_hashes": self.num_hashes,
                    "num_buckets": self.num_buckets,
                    "out_prefix": "user_",
                },
            )
            .map(
                hash_features,
                fn_kwargs={
                    "feature_names": self.item_features,
                    "num_hashes": self.num_hashes,
                    "num_buckets": self.num_buckets,
                    "out_prefix": "item_",
                },
            )
            .map(
                gather_inputs,
                fn_kwargs={
                    "user_idx": self.user_idx,
                    "item_idx": self.item_idx,
                    "label": self.label,
                    "weight": self.weight,
                },
            )
        )
        if materialize:
            ds = ds.materialize()
        return ds

    def get_dataloader(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        return dataset.iter_torch_batches(
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            local_shuffle_buffer_size=self.batch_size * 2**4,
        )

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == "fit":
            self.train_data = self.get_dataset("train", materialize=True)

        if stage in ("fit", "validate"):
            self.val_data = self.get_dataset("val", materialize=True)


if __name__ == "__main__":
    import rich

    dm = Movielens1mPipeDataModule()
    dm.prepare_data().head().collect().glimpse()
    dm.setup("fit")
    rich.print(next(iter(dm.train_dataloader())))
    rich.print(next(iter(dm.val_dataloader())))
