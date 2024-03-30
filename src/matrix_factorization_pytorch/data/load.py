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
import torch.utils.data as torch_data
import torch.utils.data._utils.collate as torch_collate
import torch.utils.data.datapipes as torch_datapipes
from filelock import FileLock

from .prepare import (
    DATA_DIR,
    MOVIELENS_1M_URL,
    download_unpack_data,
    load_dense_movielens,
)


def hash_features(
    row: dict[str, int | float | str | list[int, str]],
    *,
    idx: str,
    feature_names: list[str],
    num_hashes: int = 2,
    num_buckets: int = 2**16 + 1,
    out_prefix: str = "",
    keep_input: bool = True,
) -> torch.Tensor:
    feature_values = []
    feature_weights = []
    num_features = 0
    # categorical features
    cat_features = [
        f"{key}:{row[key]}" for key in feature_names if isinstance(row[key], (int, str))
    ]
    if len(cat_features) > 0:
        feature_values.extend(cat_features)
        feature_weights.append(torch.ones(len(cat_features)))
        num_features += len(cat_features)

    # float features
    float_features = [key for key in feature_names if isinstance(row[key], float)]
    if len(float_features) > 0:
        feature_values.extend(float_features)
        feature_weights.append(torch.as_tensor([row[key] for key in float_features]))
        num_features += len(float_features)

    # multi categorical features
    for key in feature_names:
        if isinstance(row[key], Iterable) and not isinstance(row[key], str):
            iter_feature_values = [
                f"{key}:{value}" for value in row[key] if isinstance(value, (int, str))
            ]
            num_values = len(iter_feature_values)
            if num_values == 0:
                continue
            iter_feature_weights = torch.ones(num_values) / num_values

            feature_values.extend(iter_feature_values)
            feature_weights.append(iter_feature_weights)
            num_features += 1

    feature_hashes = [
        mmh3.hash(values, seed)
        for seed in range(num_hashes)
        for values in feature_values
    ]
    feature_hashes = torch.as_tensor(feature_hashes) % (num_buckets - 1) + 1
    feature_weights = (
        torch.cat(feature_weights).tile(num_hashes) / num_features / num_hashes
    )

    new_row = {
        f"{out_prefix}idx": row[idx] or 0,
        f"{out_prefix}feature_hashes": feature_hashes.to(torch.int32),
        f"{out_prefix}feature_weights": feature_weights.to(torch.float32),
    }
    if not keep_input:
        return new_row
    row.update(new_row)
    return row


def gather_inputs(
    row: dict[str, int | float | list[int | float] | None],
    *,
    label: str = "label",
    weight: str = "weight",
) -> dict[str, int | float | list[int | float]]:
    label_value = row[label] or 0
    inputs = {
        "label": bool(label_value > 0) - bool(label_value < 0),
        "weight": row[weight] or 0,
        "user_idx": row["user_idx"],
        "user_feature_hashes": row["user_feature_hashes"],
        "user_feature_weights": row["user_feature_weights"],
        "item_idx": row["item_idx"],
        "item_feature_hashes": row["item_feature_hashes"],
        "item_feature_weights": row["item_feature_weights"],
    }
    return inputs


def merge_rows(rows):
    new_row = {**rows[0]}
    for row in rows[1:]:
        new_row = {**new_row, **row}
    return new_row


def collate_tensor_fn(
    batch: Iterable[torch.Tensor], *, collate_fn_map: dict | None = None
) -> torch.Tensor:
    it = iter(batch)
    elem_size = next(it).size()
    if (
        any(elem.size() != elem_size for elem in it)
        and (
            # only last dimension different
            all(elem.size()[:-1] == elem_size[:-1] for elem in it)
            # only first dimension differen
            or all(elem.size()[1:] == elem_size[1:] for elem in it)
        )
    ):
        # pad tensor if only first or last dimensions are different
        nested = torch.nested.as_nested_tensor(batch)
        return torch.nested.to_padded_tensor(nested, padding=0)
    return torch_collate.collate_tensor_fn(batch, collate_fn_map=collate_fn_map)


torch_collate.default_collate_fn_map[torch.Tensor] = collate_tensor_fn


def ray_collate_fn(
    batch: np.ndarray | dict[str, np.ndarray],
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return {
            col_name: torch_data.default_collate(col_batch)
            for col_name, col_batch in batch.items()
        }

    return torch_data.default_collate(batch)


@torch_data.functional_datapipe("load_pyarrow_dataset_as_dict")
class PyArrowDatasetDictLoaderIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self,
        source_datapipe: Iterable[str | Path],
        *,
        columns: Iterable[str] | None = None,
        filter_expr: ds.Expression | None = None,
        batch_size: int = 2**10,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.filter_expr = filter_expr
        self.batch_size = batch_size

    def __len__(self) -> int:
        num_rows = sum(
            ds.dataset(source).count_rows(filter=self.filter_expr)
            for source in self.source_datapipe
        )
        return num_rows

    def __iter__(self) -> Iterable[dict]:
        for source in self.source_datapipe:
            dataset = ds.dataset(source)
            for batch in dataset.to_batches(
                columns=self.columns,
                filter=self.filter_expr,
                batch_size=self.batch_size,
            ):
                yield from batch.to_pylist()


@torch_data.functional_datapipe("cycle")
class CyclerIterDataPipe(torch_data.IterDataPipe):
    def __init__(
        self,
        source_datapipe: Iterable[str | Path],
        count: int | None = None,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.count = count
        if count is not None and count < 0:
            msg = f"requires count >= 0, but {count = }"
            raise ValueError(msg)

    def __len__(self) -> int:
        if self.count is None:
            # use arbitrary large number so that valid length is shown for zip
            return 2**32
        return len(self.source_datapipe) * self.count

    def __iter__(self) -> Iterable[dict]:
        i = 0
        while self.count is None or i < self.count:
            yield from self.source_datapipe
            i += 1


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
        self,
        data_dir: str = DATA_DIR,
        num_hashes: int = 2,
        num_buckets: int = 2**16 + 1,
        batch_size: int = 2**10,
        negative_multiple: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self, *, overwrite: bool = False):
        with FileLock(f"{self.hparams.data_dir}.lock"):
            download_unpack_data(self.url, self.hparams.data_dir, overwrite=overwrite)
            return load_dense_movielens(self.hparams.data_dir, overwrite=overwrite)

    @abc.abstractclassmethod
    def get_dataset(self, subset: str) -> torch.utils.data.Dataset: ...

    @abc.abstractmethod
    def get_dataloader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader: ...

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
    def get_movies_dataset(self, cycle_count: int | None = 1):
        parquet_path = Path(self.hparams.data_dir, "ml-1m", "movies.parquet")

        datapipe = (
            torch_datapipes.iter.IterableWrapper([parquet_path])
            .load_pyarrow_dataset_as_dict(
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
                    out_prefix="neg_item_",
                    keep_input=False,
                )
            )
        )
        return datapipe

    def get_dataset(self, subset: str) -> torch.utils.data.Dataset:
        assert subset in {"train", "val", "test", "predict"}
        parquet_file = {
            "train": "sparse.parquet",
            "val": "val.parquet",
        }.get(subset, "dense.parquet")
        parquet_path = Path(self.hparams.data_dir, "ml-1m", parquet_file)
        filter_col = f"is_{subset}"

        datapipe = (
            torch_datapipes.iter.IterableWrapper([parquet_path])
            .load_pyarrow_dataset_as_dict(
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
        if subset == "train" and self.hparams.negative_multiple > 0:
            datapipe = (
                self.get_movies_dataset(cycle_count=None)
                .batch(self.hparams.negative_multiple)
                .collate()
                .zip(datapipe)
                .map(merge_rows)
            )

        return datapipe

    def get_dataloader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() - 1,
            persistent_workers=True,
        )


class Movielens1mRayDataModule(Movielens1mBaseDataModule):
    def get_dataset(self, subset: str) -> ray.data.Dataset:
        assert subset in {"train", "val", "test", "predict"}
        parquet_file = {
            "train": "sparse.parquet",
            "val": "val.parquet",
        }.get(subset, "dense.parquet")
        parquet_path = Path(self.hparams.data_dir, "ml-1m", parquet_file)
        filter_col = f"is_{subset}"

        dataset = (
            ray.data.read_parquet(
                parquet_path,
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
        return dataset

    def get_dataloader(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
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
