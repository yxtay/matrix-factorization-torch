from __future__ import annotations

import datetime
import functools
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pydantic
import torch
import torch.utils.data as torch_data
from lightning import LightningDataModule

from mf_torch.data.load import collate_features, hash_features, select_fields
from mf_torch.params import (
    BATCH_SIZE,
    DATA_DIR,
    ITEM_FEATURE_NAMES,
    ITEM_IDX,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MOVIELENS_1M_URL,
    NUM_EMBEDDINGS,
    NUM_HASHES,
    TARGET,
    USER_FEATURE_NAMES,
    USER_IDX,
    USERS_TABLE_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Self, TypeVar

    import lancedb
    import polars as pl

    T = TypeVar("T")


FEATURES_TYPE = dict[str, torch.Tensor]
BATCH_TYPE = tuple[FEATURES_TYPE, FEATURES_TYPE, torch.Tensor]


class FeaturesProcessor(pydantic.BaseModel):
    data_dir: str = DATA_DIR
    idx_col: str
    feature_names: dict[str, str]
    table_name: str

    lance_db_path: str = LANCE_DB_PATH
    batch_size: int = BATCH_SIZE
    num_hashes: int = NUM_HASHES
    num_embeddings: int = NUM_EMBEDDINGS

    def process(self: Self, example: dict[str, Any]) -> dict[str, Any]:
        features = select_fields(example, fields=list(self.feature_names))
        feature_values, feature_weights = collate_features(
            features, feature_names=self.feature_names
        )
        feature_hashes, feature_weights = hash_features(
            feature_values,
            feature_weights,
            num_hashes=self.num_hashes,
            num_embeddings=self.num_embeddings,
        )

        return {
            **example,
            "idx": example[self.idx_col],
            "feature_values": feature_values,
            "feature_hashes": feature_hashes,
            "feature_weights": feature_weights,
        }

    def get_data(
        self: Self, subset: str = "train"
    ) -> torch_data.IterDataPipe[FEATURES_TYPE]:
        import pyarrow.dataset as ds

        from mf_torch.data.load import ParquetDictLoaderIterDataPipe

        valid_subset = {"train", "val", "test", "predict"}
        if subset not in valid_subset:
            msg = f"`{subset}` is not one of `{valid_subset}`"
            raise ValueError(msg)

        filter_expr = ds.field(f"is_{subset}")
        return ParquetDictLoaderIterDataPipe(
            [self.data_path], filter_expr=filter_expr, batch_size=self.batch_size
        ).map(self.process)

    def get_train_data(
        self: Self,
    ) -> torch_data.IterDataPipe[FEATURES_TYPE]:
        import torch.utils.data._utils.collate as torch_collate

        fields = ["idx", "feature_hashes", "feature_weights"]
        return (
            self.get_data("train")
            .shuffle(buffer_size=self.batch_size)  # devskim: ignore DS148264
            .sharding_filter()
            .map(functools.partial(select_fields, fields=fields))
            .batch(self.batch_size)
            .map(torch_collate.default_collate)
        )


class UsersProcessor(FeaturesProcessor):
    idx_col: str = USER_IDX
    feature_names: dict[str, str] = USER_FEATURE_NAMES
    table_name: str = USERS_TABLE_NAME

    @property
    def data_path(self) -> str:
        return str(Path(self.data_dir, "ml-1m", "users.parquet"))

    def get_index(self: Self) -> lancedb.table.Table:
        import lancedb
        import pyarrow.parquet as pq

        columns = [
            "user_id",
            "gender",
            "age",
            "occupation",
            "zipcode",
            "history",
            "target",
        ]
        pa_table = pq.read_table(self.data_path, columns=columns)

        db = lancedb.connect(self.lance_db_path)
        table = db.create_table(self.table_name, data=pa_table, mode="overwrite")
        table.compact_files()
        table.cleanup_old_versions(datetime.timedelta(days=1))
        return table


class ItemsProcessor(FeaturesProcessor):
    idx_col: str = ITEM_IDX
    feature_names: dict[str, str] = ITEM_FEATURE_NAMES
    table_name: str = ITEMS_TABLE_NAME

    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    num_probes: int | None = None

    @property
    def data_path(self) -> str:
        return str(Path(self.data_dir, "ml-1m", "movies.parquet"))

    def get_index(self: Self, model: torch.nn.Module) -> lancedb.table.Table:
        import lancedb
        import pyarrow as pa

        from mf_torch.data.load import ParquetDictLoaderIterDataPipe

        fields = ["movie_id", "title", "genres", "embedding"]
        dp = (
            ParquetDictLoaderIterDataPipe([self.data_path])
            .map(self.process)
            .map(
                lambda example: {
                    **example,
                    "embedding": model(
                        example["feature_hashes"].unsqueeze(0),
                        example["feature_weights"].unsqueeze(0),
                    )
                    .squeeze(0)
                    .numpy(force=True),
                }
            )
            .map(functools.partial(select_fields, fields=fields))
        )

        example = next(iter(dp))
        num_items = len(dp)
        (embedding_dim,) = example["embedding"].shape
        # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
        num_partitions = self.num_partitions or 2 ** int(math.log2(num_items) / 2)
        num_sub_vectors = self.num_sub_vectors or embedding_dim // 8

        schema = pa.RecordBatch.from_pylist([example]).schema
        schema = schema.set(
            schema.get_field_index("embedding"),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        )
        pa_table = pa.Table.from_pylist(list(dp), schema=schema)

        db = lancedb.connect(self.lance_db_path)
        table = db.create_table(
            self.table_name,
            data=pa_table,
            mode="overwrite",
        )
        table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="embedding",
        )
        table.compact_files()
        table.cleanup_old_versions(datetime.timedelta(days=1))
        return table


class MatrixFactorisationDataPipe(torch_data.IterDataPipe[BATCH_TYPE]):
    def __init__(
        self: Self,
        users_dataset: torch_data.IterDataPipe[FEATURES_TYPE],
        items_dataset: torch_data.IterDataPipe[FEATURES_TYPE],
        data_dir: str = DATA_DIR,
    ) -> None:
        super().__init__()
        self.users_dataset = users_dataset
        self.items_dataset = items_dataset
        self.data_path = str(Path(data_dir, "ml-1m", "ratings.parquet"))
        self.targets = self.get_targets_matrix()

    def get_targets_matrix(self: Self) -> torch.Tensor:
        import polars as pl
        import scipy

        columns = {TARGET, USER_IDX, ITEM_IDX}
        ratings_df = (
            pl.scan_parquet(self.data_path)
            .filter(pl.col("is_train"))
            .select(columns)
            .collect()
            .to_pandas()
        )
        values = ratings_df[TARGET]
        rows = ratings_df[USER_IDX]
        cols = ratings_df[ITEM_IDX]
        return scipy.sparse.coo_array((values, (rows, cols))).tocsr()

    def __len__(self: Self) -> int:
        return len(self.users_dataset) * len(self.items_dataset)

    def __iter__(
        self: Self,
    ) -> Iterator[BATCH_TYPE]:
        for users_batch in iter(self.users_dataset):
            for items_batch in iter(self.items_dataset):
                row_idx = users_batch["idx"].numpy(force=True)
                col_idx = items_batch["idx"].numpy(force=True)
                targets = self.targets[row_idx[:, None], col_idx[None, :]]
                targets_batch = torch.sparse_csr_tensor(
                    targets.indptr,
                    targets.indices,
                    targets.data,
                    targets.shape,
                )
                yield {
                    "targets": targets_batch,
                    "users": users_batch,
                    "items": items_batch,
                }


class MatrixFactorizationDataModule(LightningDataModule):
    def __init__(
        self: Self,
        data_dir: str = DATA_DIR,
        user_batch_size: int = BATCH_SIZE,
        item_batch_size: int = BATCH_SIZE,
        num_hashes: int = NUM_HASHES,
        num_embeddings: int = NUM_EMBEDDINGS,
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
        num_probes: int | None = None,
        num_workers: int | None = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self: Self, *, overwrite: bool = False) -> pl.LazyFrame:
        from filelock import FileLock

        from mf_torch.data.prepare import download_unpack_data, prepare_movielens

        with FileLock(f"{self.hparams.data_dir}.lock"):
            download_unpack_data(
                MOVIELENS_1M_URL, self.hparams.data_dir, overwrite=overwrite
            )
            return prepare_movielens(self.hparams.data_dir, overwrite=overwrite)

    def setup(self: Self, *args) -> None:
        self.users_processor = UsersProcessor(
            batch_size=self.hparams.user_batch_size,
            num_hashes=self.hparams.num_hashes,
            num_embeddings=self.hparams.num_embeddings,
        )
        self.items_processor = ItemsProcessor(
            batch_size=self.hparams.item_batch_size,
            num_hashes=self.hparams.num_hashes,
            num_embeddings=self.hparams.num_embeddings,
            num_partitions=self.hparams.num_partitions,
            num_sub_vectors=self.hparams.num_sub_vectors,
            num_probes=self.hparams.num_probes,
        )

    def get_dataloader(
        self: Self,
        dataset: torch_data.Dataset[T],
        *,
        shuffle: bool = True,
    ) -> torch_data.DataLoader[T]:
        num_workers = self.hparams.get("num_workers")

        if num_workers is None:
            if cpu_count := os.cpu_count() is not None:
                num_workers = cpu_count - 1
            else:
                num_workers = 1

        multiprocessing_context = "spawn" if num_workers > 0 else None

        return torch_data.DataLoader(
            dataset,
            batch_size=None,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self: Self) -> torch_data.DataLoader[BATCH_TYPE]:
        train_data = MatrixFactorisationDataPipe(
            users_dataset=self.users_processor.get_train_data(),
            items_dataset=self.items_processor.get_train_data(),
        ).shuffle(buffer_size=2**4)  # devskim: ignore DS148264
        return self.get_dataloader(train_data)

    def val_dataloader(self: Self) -> torch_data.DataLoader[FEATURES_TYPE]:
        val_data = self.users_processor.get_data("val")
        return self.get_dataloader(val_data, shuffle=False)

    def test_dataloader(self: Self) -> torch_data.DataLoader[FEATURES_TYPE]:
        test_data = self.users_processor.get_data("test")
        return self.get_dataloader(test_data, shuffle=False)

    def predict_dataloader(self: Self) -> torch_data.DataLoader[FEATURES_TYPE]:
        predict_data = self.users_processor.get_data("predict")
        return self.get_dataloader(predict_data, shuffle=False)


if __name__ == "__main__":
    import rich.pretty

    dm = MatrixFactorizationDataModule()
    dm.prepare_data().head().collect().glimpse()
    dm.setup()

    dataloaders = [
        dm.users_processor.get_data(),
        dm.items_processor.get_data(),
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

    dm.users_processor.get_index().search().to_polars().glimpse()
    dm.items_processor.get_index(
        lambda hashes, _: torch.rand(hashes.size(0), 32)  # devskim: ignore DS148264
    ).search().to_polars().glimpse()
