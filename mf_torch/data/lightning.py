from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pydantic
import torch
import torch.utils.data as torch_data
import torch.utils.data.datapipes as torch_datapipes
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
    USER_FEATURE_NAMES,
    USER_IDX,
    USERS_TABLE_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, ClassVar, Self

    import lancedb
    import polars as pl


LABEL = "rating"
WEIGHT = "rating"


class FeatureProcessor(pydantic.BaseModel):
    data_path: str
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
        self: Self, subset: str = "val"
    ) -> torch_data.IterDataPipe[dict[str, torch.Tensor]]:
        import pyarrow.dataset as ds

        assert subset in {"train", "val", "test", "predict"}
        filter_expr = ds.field(f"is_{subset}")
        return (
            torch_datapipes.iter.IterableWrapper([self.data_path])
            .load_parquet_as_dict(filter_expr=filter_expr, batch_size=self.batch_size)
            .sharding_filter()
            .map(self.process)
        )

    def get_train_data(
        self: Self,
    ) -> torch_data.IterDataPipe[dict[str, torch.Tensor]]:
        import pyarrow.dataset as ds
        import torch.utils.data._utils.collate as torch_collate

        fields = ["idx", "feature_hashes", "feature_weights"]
        return (
            torch_datapipes.iter.IterableWrapper([self.data_path])
            .load_parquet_as_dict(
                filter_expr=ds.field("is_train"), batch_size=self.batch_size
            )
            .shuffle(buffer_size=self.batch_size)
            .map(self.process)
            .map(functools.partial(select_fields, fields=fields))
            .batch(self.batch_size)
            .map(torch_collate.default_collate)
        )


class UsersProcessor(FeatureProcessor):
    data_path: str = str(Path(DATA_DIR, "ml-1m", "users.parquet"))
    idx_col: str = USER_IDX
    feature_names: dict[str, str] = USER_FEATURE_NAMES
    table_name: str = USERS_TABLE_NAME

    def get_index(self: Self) -> lancedb.table.Table:
        import datetime

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


class ItemsProcessor(FeatureProcessor):
    data_path: str = str(Path(DATA_DIR, "ml-1m", "movies.parquet"))
    idx_col: str = ITEM_IDX
    feature_names: dict[str, str] = ITEM_FEATURE_NAMES
    table_name: str = ITEMS_TABLE_NAME

    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    nprobe: int | None = None

    def get_index(self: Self, model: torch.nn.Module) -> lancedb.table.Table:
        import datetime

        import lancedb
        import pyarrow as pa

        fields = ["movie_id", "title", "genres", "embedding"]
        dp = (
            torch_datapipes.iter.IterableWrapper([self.data_path])
            .load_parquet_as_dict()
            .sharding_filter()
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


class MatrixFactorisationDataPipe(
    torch_data.IterDataPipe[
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    ]
):
    def __init__(
        self: Self,
        users_dataset: torch_data.IterDataPipe[dict[str, torch.Tensor]],
        items_dataset: torch_data.IterDataPipe[dict[str, torch.Tensor]],
        data_path: str = str(Path(DATA_DIR, "ml-1m", "ratings.parquet")),
    ) -> None:
        super().__init__()
        self.users_dataset = users_dataset
        self.items_dataset = items_dataset
        self.data_path = data_path
        self.ratings = self.get_ratings_matrix()

    def get_ratings_matrix(self: Self) -> torch.Tensor:
        import polars as pl
        import scipy

        columns = {WEIGHT, USER_IDX, ITEM_IDX}
        ratings_df = (
            pl.scan_parquet(self.data_path)
            .filter(pl.col("is_train"))
            .select(columns)
            .collect()
            .to_pandas()
        )
        values = ratings_df["rating"]
        rows = ratings_df[USER_IDX]
        cols = ratings_df[ITEM_IDX]
        return scipy.sparse.coo_array((values, (rows, cols))).tocsr()

    def __len__(self: Self) -> int:
        return len(self.users_dataset) * len(self.items_dataset)

    def __iter__(
        self: Self,
    ) -> Iterator[
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    ]:
        for users_batch in iter(self.users_dataset):
            for items_batch in iter(self.items_dataset):
                row_idx = users_batch["idx"].numpy(force=True)
                col_idx = items_batch["idx"].numpy(force=True)
                ratings = self.ratings[row_idx[:, None], col_idx[None, :]]
                ratings_batch = torch.sparse_csr_tensor(
                    ratings.indptr,
                    ratings.indices,
                    ratings.data,
                    ratings.shape,
                )
                yield {
                    "ratings": ratings_batch,
                    "users": users_batch,
                    "items": items_batch,
                }


class MatrixFactorizationDataModule(LightningDataModule):
    label: str = LABEL
    weight: str = WEIGHT
    user_idx: str = USER_IDX
    user_features: ClassVar[dict[str, str]] = USER_FEATURE_NAMES
    item_idx: str = ITEM_IDX
    item_features: ClassVar[dict[str, str]] = ITEM_FEATURE_NAMES

    def __init__(
        self: Self,
        data_dir: str = DATA_DIR,
        num_hashes: int = NUM_HASHES,
        num_embeddings: int = NUM_EMBEDDINGS,
        use_negatives: int = 1,
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
        parquet_path = Path(self.hparams.data_dir, "ml-1m", "movies.parquet")
        return torch_datapipes.iter.IterableWrapper(
            [parquet_path]
        ).load_parquet_as_dict(batch_size=self.hparams.batch_size)

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
                functools.partial(
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

        parquet_file = {
            "train": "sparse.parquet",
            "val": "val.parquet",
        }.get(subset, "dense.parquet")
        parquet_path = Path(self.hparams.data_dir, "ml-1m", parquet_file)

        columns = {
            self.label,
            self.weight,
            self.item_idx,
            *self.item_features,
            self.user_idx,
            *self.user_features,
        }
        return (
            torch_datapipes.iter.IterableWrapper([parquet_path])
            .load_parquet_as_dict(
                columns=list(columns),
                filter_expr=ds.field(filter_col),
                batch_size=self.hparams.batch_size,
            )
            .map(
                functools.partial(
                    score_interactions, label=self.label, weight=self.weight
                )
            )
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
                functools.partial(
                    process_features,
                    idx=self.item_idx,
                    feature_names=self.item_features,
                    prefix="item_",
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
            .map(
                functools.partial(
                    process_features,
                    idx=self.user_idx,
                    feature_names=self.user_features,
                    prefix="user_",
                    num_hashes=self.hparams.num_hashes,
                    num_embeddings=self.hparams.num_embeddings,
                )
            )
            .map(functools.partial(select_fields, fields=fields))
        )
        if subset == "train" and self.hparams.use_negatives:
            fields = [
                "neg_item_idx",
                "neg_item_feature_hashes",
                "neg_item_feature_weights",
            ]
            items_datapipe = self.get_items_dataset(
                cycle_count=None, prefix="neg_item_"
            ).map(functools.partial(select_fields, fields=fields))
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

    user_processor = UsersProcessor()
    # rich.print(next(iter(user_processor.get_train_data())))
    # rich.print(next(iter(user_processor.get_data("val"))))
    # user_processor.get_index().search().limit(5).to_polars().glimpse()
    item_processor = ItemsProcessor()
    # rich.print(next(iter(item_processor.get_train_data())))
    # (
    #     item_processor.get_index(lambda hashes, weights: torch.rand(1, 32))
    #     .search()
    #     .limit(5)
    #     .to_polars()
    #     .glimpse()
    # )
    dp = MatrixFactorisationDataPipe(
        user_processor.get_train_data(), item_processor.get_train_data()
    )
    rich.print(next(iter(dp)))

    # dm = MatrixFactorizationDataModule(num_workers=1)
    # dm.prepare_data().head().collect().glimpse()
    # dm.setup("fit")

    # dataloaders = [
    #     dm.get_items_dataset(),
    #     dm.train_data,
    #     dm.val_data,
    #     dm.train_dataloader(),
    #     dm.val_dataloader(),
    # ]
    # for dataloader in dataloaders:
    #     batch = next(iter(dataloader))
    #     rich.print(batch)
    #     shapes = {
    #         key: value.shape
    #         for key, value in batch.items()
    #         if isinstance(value, torch.Tensor)
    #     }
    #     rich.print(shapes)
