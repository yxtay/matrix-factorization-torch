from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from docarray import DocList

from mf_torch.bentoml.schemas import MovieCandidate, MovieSchema, Query
from mf_torch.params import (
    EMBEDDER_PATH,
    LANCE_DB_PATH,
    MODEL_NAME,
    MOVIES_DOC_PATH,
    MOVIES_TABLE_NAME,
)

if TYPE_CHECKING:
    import lancedb.table
    from mf_torch.lightning import LitMatrixFactorization


def prepare_model() -> LitMatrixFactorization:
    from mf_torch.lightning import cli_main

    cli = cli_main(["fit"])
    return cli.trainer.lightning_module.eval()


def prepare_movies() -> DocList[MovieCandidate]:
    import polars as pl

    from mf_torch.data.prepare import load_movies

    movies_df = (
        load_movies()
        .select(
            pl.col("movie_id").alias("id"),
            pl.col("movie_id"),
            pl.col("movie_idx"),
            pl.col("title"),
            pl.col("genres"),
        )
        .collect()
        .to_pandas()
    )
    movies = DocList[MovieCandidate].from_dataframe(movies_df)
    queries = DocList[Query](movie.to_query() for movie in movies)
    movies.feature_hashes = queries.feature_hashes
    movies.feature_weights = queries.feature_weights
    return movies


def embed_movies(
    movies: DocList[MovieCandidate], model: LitMatrixFactorization
) -> DocList[MovieCandidate]:
    with torch.inference_mode():
        feature_hashes = torch.nested.nested_tensor(
            movies.feature_hashes
        ).to_padded_tensor(padding=0)
        feature_weights = torch.nested.nested_tensor(
            movies.feature_weights
        ).to_padded_tensor(padding=0)
        embeddings = model(feature_hashes, feature_weights)
        movies.embedding = list(embeddings)
    return movies


def prepare_index(movies: DocList[MovieCandidate]) -> lancedb.table.LanceTable:
    import datetime

    import lancedb

    num_partitions = int(len(movies) ** 0.5)
    num_sub_vectors = int(movies[0].embedding.size / 8)

    db = lancedb.connect(LANCE_DB_PATH)
    table = db.create_table(
        MOVIES_TABLE_NAME,
        movies.to_dataframe(),
        mode="overwrite",
        schema=MovieSchema,
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


def save_model(movies: DocList[MovieCandidate], model: LitMatrixFactorization) -> None:
    import shutil

    import bentoml

    with bentoml.models.create(MODEL_NAME) as model_ref:
        shutil.copytree(LANCE_DB_PATH, model_ref.path_of(LANCE_DB_PATH))
        movies.push(Path(model_ref.path_of(MOVIES_DOC_PATH)).as_uri())
        model.save_torchscript(model_ref.path_of(EMBEDDER_PATH))


def load_embedded_movies() -> DocList[MovieCandidate]:
    import bentoml

    model_ref = bentoml.models.get(MODEL_NAME)
    path = Path(model_ref.path_of(MOVIES_DOC_PATH)).as_uri()
    return DocList[MovieCandidate].pull(path)


def main() -> None:
    model = prepare_model()
    movies = prepare_movies()
    movies = embed_movies(movies=movies, model=model)
    prepare_index(movies=movies)
    save_model(movies=movies, model=model)


if __name__ == "__main__":
    main()
