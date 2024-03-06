import concurrent.futures
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import lightning as L
import mmh3
import numpy as np
import pandas as pd
import polars as pl
import ray
import requests
import torch
from loguru import logger
from rich import print

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
USER_FEATURES = ["user_id", "gender", "age", "occupation", "zipcode"]
ITEM_FEATURES = ["movie_id", "genres"]


###
# download data
###


def download_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, overwrite: bool = False
) -> Path:
    # prepare destination
    dest = Path(dest_dir) / Path(url).name

    # downlaod zip
    if not dest.exists() or overwrite:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading data: {}", url)
        response = requests.get(url, stream=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(response.raw, f)

    logger.info("data downloaded: {}", dest)
    return dest


def unpack_data(archive_file: str, overwrite=False) -> list[str]:
    archive_file = Path(archive_file)
    dest_dir = archive_file.parent / archive_file.stem

    if not dest_dir.exists() or overwrite:
        shutil.unpack_archive(archive_file, dest_dir.parent)

    unpacked_files = [file.name for file in dest_dir.iterdir()]
    logger.info("data unpacked: {}", unpacked_files)
    return unpacked_files


def download_unpack_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, overwrite: bool = False
) -> None:
    archive_file = download_data(url=url, dest_dir=dest_dir, overwrite=overwrite)
    unpack_data(archive_file, overwrite=overwrite)


###
# load data
###


def load_users(src_dir: str = DATA_DIR, overwrite: bool = False) -> pl.LazyFrame:
    users_parquet = Path(src_dir, "ml-1m", "users.parquet")
    if users_parquet.exists() and not overwrite:
        users = pl.scan_parquet(users_parquet)
        logger.info("users loaded: {}", users_parquet)
        return users

    users_dat = Path(src_dir, "ml-1m", "users.dat")
    dtype = {
        "user_id": "int32",
        "gender": "category",
        "age": "int32",
        "occupation": "int32",
        "zipcode": "category",
    }

    users = pd.read_csv(
        users_dat,
        sep="::",
        header=None,
        names=dtype.keys(),
        dtype=dtype,
        engine="python",
    ).pipe(pl.from_pandas)
    logger.info("users loaded: {}, shape: {}", users_dat, users.shape)

    users = users.sort("user_id").with_columns(
        pl.col("user_id").rank().cast(pl.Int32).alias("user_idx")
    )
    users.write_parquet(users_parquet)
    logger.info("users saved: {}", users_parquet)

    users = pl.scan_parquet(users_parquet)
    return users


def load_movies(src_dir: str = DATA_DIR, overwrite: bool = False) -> pl.LazyFrame:
    movies_parquet = Path(src_dir, "ml-1m", "movies.parquet")
    if movies_parquet.exists() and not overwrite:
        movies = pl.scan_parquet(movies_parquet)
        logger.info("movies loaded: {}", movies_parquet)
        return movies

    movies_dat = Path(src_dir, "ml-1m", "movies.dat")
    dtype = {"movie_id": "int32", "title": "category", "genres": "str"}
    movies = pd.read_csv(
        movies_dat,
        sep="::",
        header=None,
        names=dtype.keys(),
        dtype=dtype,
        engine="python",
        encoding="iso-8859-1",
    ).pipe(pl.from_pandas)
    logger.info("movies loaded: {}, shape: {}", movies_dat, movies.shape)

    movies = movies.sort("movie_id").with_columns(
        pl.col("genres").str.split("|").cast(pl.List(pl.Categorical)).alias("genres"),
        pl.col("movie_id").rank().cast(pl.Int32).alias("movie_idx"),
    )
    movies.write_parquet(movies_parquet)
    logger.info("movies saved: {}", movies_parquet)

    movies = pl.scan_parquet(movies_parquet)
    return movies


def load_ratings(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    ratings_dat = Path(src_dir, "ml-1m", "ratings.dat")
    dtype = {
        "user_id": "int32",
        "movie_id": "int32",
        "rating": "int32",
        "timestamp": "int32",
    }
    ratings = pd.read_csv(
        ratings_dat,
        sep="::",
        header=None,
        names=dtype.keys(),
        dtype=dtype,
        engine="python",
    ).pipe(pl.from_pandas)
    logger.info("ratings loaded: {}, shape: {}", ratings_dat, ratings.shape)
    return ratings.lazy()


###
# process data
###


def ordered_split(
    data: pl.LazyFrame,
    *,
    group_col: str = "user_id",
    order_col: str = "timestamp",
    valid_prop: float = 0.1,
    test_prop: float = 0.1,
) -> pl.LazyFrame:
    valid_threshold = 1 - test_prop
    train_threshold = valid_threshold - valid_prop

    data = (
        data.lazy()
        .with_columns(
            p=(pl.col(order_col).rank("ordinal") / pl.col(order_col).count()).over(
                group_col
            )
        )
        .with_columns(
            is_rated=True,
            is_train=pl.col("p") <= train_threshold,
            is_val=(train_threshold < pl.col("p")) & (pl.col("p") <= valid_threshold),
            is_test=valid_threshold < pl.col("p"),
        )
        .drop("p")
    )
    return data


def get_dense_interactions(
    users: pl.LazyFrame, movies: pl.LazyFrame, data: pl.LazyFrame, parquet_path: str
) -> pl.LazyFrame:
    dense_interactions = (
        users.lazy()
        .join(movies.lazy(), how="cross")
        .join(data.lazy(), on=["user_id", "movie_id"], how="left")
        .with_columns(
            is_rated=pl.col("is_rated").fill_null(False),
            is_train=pl.col("is_train").fill_null(False),
            is_val=pl.col("is_val").fill_null(True),
            is_test=pl.col("is_test").fill_null(True),
            is_predict=True,
        )
    )
    test_val = (
        # filter interactions that are in test only
        dense_interactions.filter(pl.col("is_rated") & pl.col("is_test")).with_columns(
            rating=None,
            timestamp=None,
            is_rated=False,
            is_train=False,
            is_val=True,
            is_test=False,
            is_predict=False,
        )
    )
    dense_interactions = pl.concat([dense_interactions, test_val]).collect(
        streaming=True
    )
    dense_interactions.write_parquet(parquet_path)
    logger.info("slice saved: {}, shape: {}", parquet_path, dense_interactions.shape)

    dense_interactions = pl.scan_parquet(parquet_path)
    return dense_interactions


def get_sparse_movielens(
    data: pl.LazyFrame, src_dir: str = DATA_DIR, overwrite: bool = False
) -> pl.LazyFrame:
    sparse_parquet = Path(src_dir, "ml-1m", "sparse.parquet")
    if sparse_parquet.exists() and not overwrite:
        sparse = pl.scan_parquet(sparse_parquet)
        logger.info("sparse loaded: {}", sparse_parquet)
        return sparse

    sparse = data.filter(pl.col("is_rated")).collect(streaming=True)
    sparse.write_parquet(sparse_parquet)
    logger.info("sparse saved: {}, shape: {}", sparse_parquet, sparse.shape)

    sparse = pl.scan_parquet(sparse_parquet)
    return sparse


def load_dense_movielens(
    src_dir: str = DATA_DIR, overwrite: bool = False
) -> pl.LazyFrame:
    dense_parquet = Path(src_dir, "ml-1m", "dense.parquet")
    if dense_parquet.exists() and not overwrite:
        dense = pl.scan_parquet(dense_parquet)
        logger.info("dense loaded: {}", dense_parquet)
        return dense

    users = load_users(src_dir, overwrite=overwrite).collect(streaming=True)
    movies = load_movies(src_dir, overwrite=overwrite)
    ratings = load_ratings(src_dir).pipe(ordered_split)

    with tempfile.TemporaryDirectory() as temp_dir, concurrent.futures.ThreadPoolExecutor() as executor:
        slice_futures = [
            executor.submit(
                get_dense_interactions,
                users_slice,
                movies,
                ratings,
                Path(temp_dir, f"part-{idx}.parquet"),
            )
            for idx, users_slice in enumerate(users.iter_slices(users.height // 10))
        ]
        dense = (
            pl.concat(
                [
                    futures.result()
                    for futures in concurrent.futures.as_completed(slice_futures)
                ]
            )
            .sort(["timestamp", "user_id", "movie_id"])
            .collect(streaming=True)
            .sample(fraction=1.0, shuffle=True, seed=0)
        )

    dense.write_parquet(dense_parquet)
    logger.info("dense saved: {}, shape: {}", dense_parquet, dense.shape)

    dense = pl.scan_parquet(dense_parquet)
    get_sparse_movielens(dense, src_dir=src_dir, overwrite=overwrite)
    return dense


###
# dataloader
###


def hash_features(
    row: dict[str, float | str | list[float | str]], feature_names: list[str]
) -> torch.Tensor:
    hashed_features = []
    for key in feature_names:
        if isinstance(row[key], Iterable) and not isinstance(row[key], str):
            hashed = [
                mmh3.hash(f"{key}:{value}") if value is not None else 0
                for value in row[key]
            ]
        else:
            hashed = [mmh3.hash(f"{key}:{row[key]}") if row[key] is not None else 0]

        hashed_features.append(torch.as_tensor(hashed, dtype=torch.int))

    return torch.nn.utils.rnn.pad_sequence(hashed_features, batch_first=True)


def gather_inputs(
    row: dict[str, bool | float | str | list[bool | float | str]],
    *,
    user_idx: str = "user_idx",
    item_idx: str = "movie_idx",
    label: str = "label",
    weight: str = "weight",
    user_features: list[str] = [],
    item_features: list[str] = [],
) -> dict[str, float | torch.Tensor]:
    new_row = {
        "user_idx": row[user_idx],
        "item_idx": row[item_idx],
        "label": row[label],
        "weight": row[weight],
        "user_features": hash_features(row, user_features),
        "item_features": hash_features(row, item_features),
    }
    return new_row


def collate_fn(
    batch: np.ndarray | dict[str, np.ndarray],
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return {
            col_name: collate_fn(col_batch) for col_name, col_batch in batch.items()
        }

    # batch is np.ndarray
    if batch.dtype.type is np.object_ and isinstance(batch[0], np.ndarray):
        # variable sequence lengths must be dim 1 in batch_first pad_sequence
        # hence transpose first, pad_sequence, then transpose back
        batch_seq_feat = [torch.as_tensor(arr.T) for arr in batch]
        padded = torch.nn.utils.rnn.pad_sequence(batch_seq_feat, batch_first=True)
        return padded.transpose(1, 2)

    return torch.as_tensor(batch)


def get_parquet_dataloader(
    parquet_file: str,
    filter_col: str,
    *,
    user_idx: str = "user_idx",
    item_idx: str = "item_idx",
    label: str = "label",
    weight: str = "weight",
    user_features: list[str] = [],
    item_features: list[str] = [],
    batch_size: int = 1024,
):
    return (
        ray.data.read_parquet(parquet_file)
        .filter(lambda row: row[filter_col])
        .map(
            gather_inputs,
            fn_kwargs={
                "user_idx": user_idx,
                "item_idx": item_idx,
                "label": label,
                "weight": weight,
                "user_features": user_features,
                "item_features": item_features,
            },
        )
        .iter_torch_batches(
            batch_size=batch_size,
            collate_fn=collate_fn,
            local_shuffle_buffer_size=batch_size * 16,
        )
    )


class Movielens1mDataModule(L.LightningDataModule):
    url: str = MOVIELENS_1M_URL
    user_idx: str = "user_idx"
    item_idx: str = "movie_idx"
    label: str = "rating"
    weight: str = "rating"
    user_features: list[str] = USER_FEATURES
    item_features: list[str] = ITEM_FEATURES

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        batch_size: int = 1024,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        download_unpack_data(self.url, self.data_dir)
        load_dense_movielens(self.data_dir)

    def dataloader(self, data_subset) -> ray.data.Dataset:
        assert data_subset in {"train", "val", "test", "predict"}
        parquet_file = Path(
            self.data_dir,
            "ml-1m",
            "sparse.parquet" if data_subset == "train" else "dense.parquet",
        )
        filter_col = f"is_{data_subset}"
        return get_parquet_dataloader(
            parquet_file=parquet_file,
            filter_col=filter_col,
            user_idx=self.user_idx,
            item_idx=self.item_idx,
            label=self.label,
            weight=self.weight,
            user_features=self.user_features,
            item_features=self.item_features,
            batch_size=self.batch_size,
        )

    def train_dataloader(self) -> ray.data.Dataset:
        return self.dataloader("train")

    def val_dataloader(self) -> ray.data.Dataset:
        return self.dataloader("val")

    def test_dataloader(self) -> ray.data.Dataset:
        return self.dataloader("test")

    def predict_dataloader(self) -> ray.data.Dataset:
        return self.dataloader("predict")


if __name__ == "__main__":
    dm = Movielens1mDataModule(batch_size=2)
    dm.prepare_data()

    load_dense_movielens().head().collect().glimpse()

    print(next(iter(dm.train_dataloader())))
    print(next(iter(dm.val_dataloader())))
