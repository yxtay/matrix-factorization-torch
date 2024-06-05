from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    import deltalake

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"


###
# download data
###


def download_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> Path:
    import requests

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


def unpack_data(archive_file: str, *, overwrite: bool = False) -> list[str]:
    archive_file = Path(archive_file)
    dest_dir = archive_file.parent / archive_file.stem

    if not dest_dir.exists() or overwrite:
        shutil.unpack_archive(archive_file, dest_dir.parent)

    unpacked_files = [file.name for file in dest_dir.iterdir()]
    logger.info("data unpacked: {}", unpacked_files)
    return unpacked_files


def download_unpack_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> list[str]:
    archive_file = download_data(url=url, dest_dir=dest_dir, overwrite=overwrite)
    return unpack_data(archive_file, overwrite=overwrite)


###
# load data
###


def optimize_delta_table(
    delta_path: str,
    columns: list[str] | None = None,
    *,
    retention_hours: int = 0,
    **kwargs,
) -> deltalake.DeltaTable:
    import deltalake

    table = deltalake.DeltaTable(delta_path)
    if columns:
        table.optimize.z_order(columns, **kwargs)
    else:
        table.optimize.compact(**kwargs)
    table.vacuum(retention_hours, dry_run=False, enforce_retention_duration=False)
    return table


def load_users(src_dir: str = DATA_DIR, *, overwrite: bool = False) -> pl.LazyFrame:
    users_delta = Path(src_dir, "ml-1m", "users.delta")
    if users_delta.exists() and not overwrite:
        users = pl.scan_delta(str(users_delta))
        logger.info("users loaded: {}", users_delta)
        return users

    import pandas as pd

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

    users = users.with_columns(
        pl.col("user_id").rank().cast(pl.Int32).alias("user_idx")
    )
    users.write_delta(
        users_delta, mode="overwrite", delta_write_options={"overwrite_schema": True}
    )
    optimize_delta_table(users_delta, ["user_id"])
    logger.info("users saved: {}", users_delta)

    return pl.scan_delta(str(users_delta))


def load_movies(src_dir: str = DATA_DIR, *, overwrite: bool = False) -> pl.LazyFrame:
    movies_delta = Path(src_dir, "ml-1m", "movies.delta")
    if movies_delta.exists() and not overwrite:
        movies = pl.scan_delta(str(movies_delta))
        logger.info("movies loaded: {}", movies_delta)
        return movies

    import pandas as pd

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

    movies = movies.with_columns(
        pl.col("genres").str.split("|").cast(pl.List(pl.Categorical)).alias("genres"),
        pl.col("movie_id").rank().cast(pl.Int32).alias("movie_idx"),
    )
    movies.write_delta(
        movies_delta, mode="overwrite", delta_write_options={"overwrite_schema": True}
    )
    optimize_delta_table(movies_delta, ["movie_id"])
    logger.info("movies saved: {}", movies_delta)

    return pl.scan_delta(str(movies_delta))


def load_ratings(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    import pandas as pd

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
    train_prop: float = 0.8,
) -> pl.LazyFrame:
    return (
        data.lazy()
        .with_columns(
            p=((pl.col(order_col).rank("ordinal") - 1) / pl.count(order_col)).over(
                group_col
            )
        )
        .with_columns(
            is_rated=True,
            is_train=pl.col("p") < train_prop,
        )
        .drop("p")
    )


def users_split_activty(
    users: pl.LazyFrame, data: pl.LazyFrame, val_prop: float = 0.02
) -> pl.LazyFrame:
    users_interactions_agg = (
        data.lazy()
        .group_by("user_id")
        .len()
        .with_columns(p=((pl.col("len").rank("ordinal") - 1) / pl.count("len")))
        .with_columns(is_val_user=pl.col("p") >= 1 - val_prop)
        .drop("len", "p")
    )
    return users.lazy().join(
        users_interactions_agg, on="user_id", how="left", coalesce=True
    )


def get_dense_interactions(
    users: pl.LazyFrame, movies: pl.LazyFrame, data: pl.LazyFrame, delta_path: str
) -> pl.LazyFrame:
    dense_interactions = (
        users.lazy()
        .join(movies.lazy(), how="cross")
        .join(data.lazy(), on=["user_id", "movie_id"], how="left", coalesce=True)
        .with_columns(
            is_rated=pl.col("is_rated").fill_null(value=False),
            is_train=pl.col("is_train").fill_null(value=False),
        )
        .with_columns(
            is_val=~pl.col("is_train") & pl.col("is_val_user"),
            is_test=~pl.col("is_train") & ~pl.col("is_val_user"),
            is_predict=True,
        )
        .collect(streaming=True)
    )
    dense_interactions.write_delta(delta_path)
    logger.info("slice saved: {}, shape: {}", delta_path, dense_interactions.shape)

    return pl.scan_delta(str(delta_path))


def get_sparse_movielens(
    data: pl.LazyFrame, src_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pl.LazyFrame:
    sparse_delta = Path(src_dir, "ml-1m", "sparse.delta")
    if sparse_delta.exists() and not overwrite:
        sparse = pl.scan_delta(str(sparse_delta))
        logger.info("sparse loaded: {}", sparse_delta)
        return sparse

    sparse = data.filter(pl.col("is_rated")).collect(streaming=True)
    sparse.write_delta(
        sparse_delta, mode="overwrite", delta_write_options={"overwrite_schema": True}
    )
    optimize_delta_table(sparse_delta, ["is_train", "is_val", "is_test"])
    logger.info("sparse saved: {}, shape: {}", sparse_delta, sparse.shape)

    return pl.scan_delta(str(sparse_delta))


def get_val_movielens(
    data: pl.LazyFrame, src_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pl.LazyFrame:
    val_delta = Path(src_dir, "ml-1m", "val.delta")
    if val_delta.exists() and not overwrite:
        val = pl.scan_delta(str(val_delta))
        logger.info("val loaded: {}", val_delta)
        return val

    val = data.filter(pl.col("is_val_user")).collect(streaming=True)
    val.write_delta(
        val_delta, mode="overwrite", delta_write_options={"overwrite_schema": True}
    )
    optimize_delta_table(val_delta, ["is_train", "is_val", "is_test"])
    logger.info("val saved: {}, shape: {}", val_delta, val.shape)

    return pl.scan_delta(str(val_delta))


def load_dense_movielens(
    src_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pl.LazyFrame:
    dense_delta = Path(src_dir, "ml-1m", "dense.delta")
    if dense_delta.exists() and not overwrite:
        dense = pl.scan_delta(str(dense_delta))
        logger.info("dense loaded: {}", dense_delta)
        return dense

    import concurrent.futures
    import tempfile

    users = load_users(src_dir, overwrite=overwrite)
    movies = load_movies(src_dir, overwrite=overwrite)
    ratings = load_ratings(src_dir).pipe(ordered_split)
    users = users_split_activty(users, ratings).collect(streaming=True)

    with tempfile.TemporaryDirectory() as temp_dir, concurrent.futures.ThreadPoolExecutor() as executor:
        slice_futures = [
            executor.submit(
                get_dense_interactions,
                users_slice,
                movies,
                ratings,
                Path(temp_dir, f"part-{idx}.delta"),
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
            .sort(["timestamp", "user_id", "movie_id"], nulls_last=True)
            .collect()
            .sample(fraction=1.0, shuffle=True, seed=0)
        )
        dense.write_delta(
            dense_delta,
            mode="overwrite",
            delta_write_options={"overwrite_schema": True},
        )
        optimize_delta_table(dense_delta, ["is_train", "is_val", "is_test"])
        logger.info("dense saved: {}, shape: {}", dense_delta, dense.shape)

    dense = pl.scan_delta(str(dense_delta))
    get_sparse_movielens(dense, src_dir=src_dir, overwrite=overwrite)
    get_val_movielens(dense, src_dir=src_dir, overwrite=overwrite)
    return dense


if __name__ == "__main__":
    download_unpack_data(overwrite=True)
    load_dense_movielens(overwrite=True).head().collect().glimpse()
