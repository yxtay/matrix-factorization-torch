from __future__ import annotations

import bentoml
import bentoml.models
import torch
from docarray import DocList
from docarray.index import InMemoryExactNNIndex
from loguru import logger

from mf_torch.bentoml.models import (
    EMBEDDER_PATH,
    LANCE_DB_PATH,
    LANCE_TABLE_NAME,
    MODEL_TAG,
    MOVIES_DOC_PATH,
    MovieCandidate,
    MovieQuery,
    Query,
    UserQuery,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.get(MODEL_TAG)

    def __init__(self: Embedder) -> None:
        try:
            path = self.model_ref.path_of(EMBEDDER_PATH)
            self.model = torch.jit.load(path)
            logger.info("embedder loaded: {}", path)
        except Exception as e:
            logger.exception(e)
            raise

    @bentoml.api(batchable=True)
    def embed(self: Embedder, queries: list[Query]) -> list[Query]:
        try:
            queries = DocList[Query](queries)
            feature_hashes = torch.nested.nested_tensor(
                queries.feature_hashes
            ).to_padded_tensor(padding=0)
            feature_weights = torch.nested.nested_tensor(
                queries.feature_weights
            ).to_padded_tensor(padding=0)

            queries.embedding = list(self.model(feature_hashes, feature_weights))
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return queries


@bentoml.service()
class DocIndex:
    model_ref = bentoml.models.get(MODEL_TAG)

    def __init__(self: DocIndex) -> None:
        from pathlib import Path

        path = Path(self.model_ref.path_of(MOVIES_DOC_PATH)).as_uri()
        doc_list = DocList[MovieCandidate].pull(path)
        logger.info("documents loaded: {}", path)
        self.doc_index = InMemoryExactNNIndex[MovieCandidate](doc_list)

    @bentoml.api(batchable=True)
    def find(self: DocIndex, queries: list[Query]) -> list[list[MovieCandidate]]:
        try:
            queries = DocList[Query](queries)
            matches, scores = self.doc_index.find_batched(
                torch.as_tensor(queries.embedding), search_field="embedding"
            )
            for i, score in enumerate(scores):
                matches[i].score = score
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return matches


@bentoml.service()
class MovieIndex:
    model_ref = bentoml.models.get(MODEL_TAG)

    def __init__(self: DocIndex) -> None:
        import lancedb

        src_path = self.model_ref.path_of(LANCE_DB_PATH)
        self.tbl = lancedb.connect(src_path).open_table(LANCE_TABLE_NAME)
        logger.info("movies index loaded: {}", src_path)

    @bentoml.api()
    def search(self: DocIndex, query: Query) -> list[MovieCandidate]:
        import polars as pl

        try:
            results_df = (
                self.tbl.search(query.embedding)
                .to_polars()
                .with_columns((1 - pl.col("_distance")).alias("score"))
                .drop("_distance")
                .to_pandas()
            )
            results = DocList[MovieCandidate].from_dataframe(results_df)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results


@bentoml.service()
class Recommender:
    embedder = bentoml.depends(Embedder)
    movie_index = bentoml.depends(MovieIndex)

    @bentoml.api()
    def recommend(self: Recommender, query: Query) -> list[MovieCandidate]:
        try:
            query = self.embedder.embed([query])[0]
            results = self.movie_index.search(query)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    def recommend_with_movie(
        self: Recommender, movie: MovieQuery
    ) -> list[MovieCandidate]:
        try:
            results = self.recommend(movie.to_query())
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    def recommend_with_user(self: Recommender, user: UserQuery) -> list[MovieCandidate]:
        try:
            results = self.recommend(user.to_query())
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results
