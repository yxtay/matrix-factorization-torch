from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
from loguru import logger

from mf_torch.params import DATA_DIR, MOVIELENS_1M_URL

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
        response = requests.get(url, timeout=10, stream=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(response.raw, f)

    logger.info("data downloaded: {}", dest)
    return dest


def unpack_data(archive_file: str | Path, *, overwrite: bool = False) -> list[str]:
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
        names=list(dtype.keys()),
        dtype=dtype,
        engine="python",
    ).pipe(pl.from_pandas)
    logger.info("ratings loaded: {}, shape: {}", ratings_dat, ratings.shape)
    return ratings.lazy()


def load_movies(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    import pandas as pd

    movies_dat = Path(src_dir, "ml-1m", "movies.dat")
    dtype = {"movie_id": "int32", "title": "category", "genres": "str"}
    movies = pd.read_csv(
        movies_dat,
        sep="::",
        header=None,
        names=list(dtype.keys()),
        dtype=dtype,
        engine="python",
        encoding="iso-8859-1",
    ).pipe(pl.from_pandas)
    logger.info("movies loaded: {}, shape: {}", movies_dat, movies.shape)

    return movies.lazy()


def load_users(src_dir: str = DATA_DIR) -> pl.LazyFrame:
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
        names=list(dtype.keys()),
        dtype=dtype,
        engine="python",
    ).pipe(pl.from_pandas)
    logger.info("users loaded: {}, shape: {}", users_dat, users.shape)
    return users.lazy()


###
# process data
###


def ordered_split(
    ratings: pl.LazyFrame,
    *,
    group_col: str = "user_id",
    order_col: str = "timestamp",
    train_prop: float = 0.8,
) -> pl.LazyFrame:
    return (
        ratings.lazy()
        .with_columns(
            p=((pl.col(order_col).rank("ordinal") - 1) / pl.count(order_col)).over(
                group_col
            )
        )
        .with_columns(is_train=pl.col("p") < train_prop)
        .drop("p")
    )


def process_users(
    users: pl.LazyFrame,
    ratings: pl.LazyFrame,
    val_prop: float = 0.1,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    users_parquet = Path(src_dir, "ml-1m", "users.parquet")
    if users_parquet.exists() and not overwrite:
        users_procesed = pl.scan_parquet(str(users_parquet))
        logger.info("users loaded: {}", users_parquet)
        return users_procesed

    users_interactions = (
        ratings.lazy()
        .group_by("user_id")
        .agg(
            pl.len(),
            pl.any("is_train").alias("is_train"),
            pl.struct("movie_id", "rating").filter(pl.col("is_train")).alias("history"),
            pl.struct("movie_id", "rating").filter(~pl.col("is_train")).alias("target"),
        )
        .with_columns(
            p=((pl.col("len").rank("ordinal") - 1) / pl.count("len")),
            history=pl.struct(
                pl.col("history")
                .list.eval(pl.element().struct.field("movie_id"))
                .alias("movie_id"),
                pl.col("history")
                .list.eval(pl.element().struct.field("rating"))
                .alias("rating"),
            ),
            target=pl.struct(
                pl.col("target")
                .list.eval(pl.element().struct.field("movie_id"))
                .alias("movie_id"),
                pl.col("target")
                .list.eval(pl.element().struct.field("rating"))
                .alias("rating"),
            ),
        )
        .with_columns(is_val=pl.col("p") >= 1 - val_prop)
        .with_columns(
            is_test=~pl.col("is_val"),
            is_predict=True,
        )
        .drop("len", "p")
    )
    users_procesed = (
        users.lazy()
        .with_row_index("user_rn")
        .join(users_interactions, on="user_id", how="left", coalesce=True)
        .collect()
    )

    users_procesed.write_parquet(users_parquet)
    logger.info("users saved: {}", users_parquet)

    return pl.scan_parquet(str(users_parquet))


def process_movies(
    movies: pl.LazyFrame,
    ratings: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    movies_parquet = Path(src_dir, "ml-1m", "movies.parquet")
    if movies_parquet.exists() and not overwrite:
        movies_processed = pl.scan_parquet(str(movies_parquet))
        logger.info("movies loaded: {}", movies_parquet)
        return movies_processed

    movies_processed = (
        movies.lazy()
        .with_columns(
            pl.col("genres")
            .str.split("|")
            .cast(pl.List(pl.Categorical))
            .alias("genres"),
        )
        .with_row_index("movie_rn")
        .join(
            ratings.lazy()
            .group_by("movie_id")
            .agg(pl.any("is_train").alias("is_train")),
            on="movie_id",
            how="left",
            coalesce=True,
        )
        .collect()
    )

    movies_processed.write_parquet(movies_parquet)
    logger.info("movies saved: {}", movies_parquet)

    return pl.scan_parquet(str(movies_parquet))


def process_ratings(
    ratings: pl.LazyFrame,
    users: pl.LazyFrame | pl.DataFrame,
    movies: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    ratings_parquet = Path(src_dir, "ml-1m", "ratings.parquet")
    if ratings_parquet.exists() and not overwrite:
        ratings_processed = pl.scan_parquet(str(ratings_parquet))
        logger.info("sparse loaded: {}", ratings_parquet)
        return ratings_processed

    ratings_processed = (
        ratings.lazy()
        .join(users.lazy().drop("is_train"), on="user_id")
        .join(movies.lazy().drop("is_train"), on="movie_id")
        .with_columns(
            is_val=~pl.col("is_train") & pl.col("is_val"),
            is_test=~pl.col("is_train") & ~pl.col("is_val"),
        )
        .collect()
    )
    ratings_processed.write_parquet(ratings_parquet)
    logger.info("sparse saved: {}, shape: {}", ratings_parquet, ratings_processed.shape)

    return pl.scan_parquet(str(ratings_parquet))


def prepare_movielens(
    src_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pl.LazyFrame:
    ratings = load_ratings(src_dir).pipe(ordered_split)
    users = load_users(src_dir)
    movies = load_movies(src_dir)

    users = process_users(users, ratings, src_dir=src_dir, overwrite=overwrite)
    movies = process_movies(movies, ratings, src_dir=src_dir, overwrite=overwrite)
    return process_ratings(ratings, users, movies, src_dir=src_dir, overwrite=overwrite)


if __name__ == "__main__":
    download_unpack_data(overwrite=True)
    prepare_movielens(DATA_DIR, overwrite=True).collect().head().glimpse()
