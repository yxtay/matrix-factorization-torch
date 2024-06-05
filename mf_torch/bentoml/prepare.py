from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from docarray import DocList

from mf_torch.bentoml.models import (
    EMBEDDER_PATH,
    LANCE_DB_PATH,
    MODEL_NAME,
    MODEL_TAG,
    MOVIES_DOC_PATH,
    MOVIES_TABLE_NAME,
    MovieCandidate,
    MovieSchema,
    Query,
)

if TYPE_CHECKING:
    import lancedb.table

    import mf_torch.lightning as mf_lightning


def prepare_model() -> mf_lightning.LitMatrixFactorization:
    import mf_torch.lightning as mf_lightning

    trainer = mf_lightning.main()
    return trainer.model


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
    movies: DocList[MovieCandidate], model: mf_lightning.LitMatrixFactorization
) -> DocList[MovieCandidate]:
    feature_hashes = torch.nested.nested_tensor(movies.feature_hashes).to_padded_tensor(
        padding=0
    )
    feature_weights = torch.nested.nested_tensor(
        movies.feature_weights
    ).to_padded_tensor(padding=0)
    embeddings = model(feature_hashes, feature_weights)
    movies.embedding = list(embeddings)
    return movies


def prepare_index(movies: DocList[MovieCandidate]) -> lancedb.table.LanceTable:
    import datetime

    import lancedb

    db = lancedb.connect(LANCE_DB_PATH)
    table = db.create_table(
        MOVIES_TABLE_NAME,
        movies.to_dataframe(),
        mode="overwrite",
        schema=MovieSchema,
    )
    table.create_index(
        metric="cosine",
        num_partitions=4,
        num_sub_vectors=4,
        vector_column_name="embedding",
    )
    table.compact_files()
    table.cleanup_old_versions(datetime.timedelta(days=1))
    return table


def save_model(
    movies: DocList[MovieCandidate], model: mf_lightning.LitMatrixFactorization
) -> None:
    import shutil

    import bentoml

    with bentoml.models.create(MODEL_NAME) as model_ref:
        shutil.copytree(LANCE_DB_PATH, model_ref.path_of(LANCE_DB_PATH))
        movies.push(Path(model_ref.path_of(MOVIES_DOC_PATH)).as_uri())
        model.save_torchscript(model_ref.path_of(EMBEDDER_PATH))


def load_embedded_movies() -> DocList[MovieCandidate]:
    import bentoml

    model_ref = bentoml.models.get(MODEL_TAG)
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
