import torch
from docarray import DocList

import mf_torch.lightning as mf_lightning
from mf_torch.bentoml.models import (
    EMBEDDER_PATH,
    MODEL_NAME,
    MOVIES_DOC_PATH,
    MovieCandidate,
    Query,
)


def prepare_model() -> mf_lightning.LitMatrixFactorization:
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


def save_model(
    movies: DocList[MovieCandidate], model: mf_lightning.LitMatrixFactorization
) -> None:
    from pathlib import Path

    import bentoml

    with bentoml.models.create(name=MODEL_NAME) as model_ref:
        movies.push(Path(model_ref.path_of(MOVIES_DOC_PATH)).as_uri())
        model.save_torchscript(model_ref.path_of(EMBEDDER_PATH))


def main() -> None:
    model = prepare_model()
    movies = prepare_movies()
    movies = embed_movies(movies=movies, model=model)
    save_model(movies=movies, model=model)


if __name__ == "__main__":
    main()
