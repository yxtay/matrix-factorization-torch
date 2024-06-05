from __future__ import annotations

import bentoml
import torch
from docarray import DocList
from loguru import logger

from mf_torch.bentoml.models import (
    EMBEDDER_PATH,
    LANCE_DB_PATH,
    MODEL_TAG,
    MOVIES_TABLE_NAME,
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
class MovieIndex:
    model_ref = bentoml.models.get(MODEL_TAG)

    def __init__(self: MovieIndex) -> None:
        import lancedb

        src_path = self.model_ref.path_of(LANCE_DB_PATH)
        self.tbl = lancedb.connect(src_path).open_table(MOVIES_TABLE_NAME)
        logger.info("movies index loaded: {}", src_path)

    @bentoml.api()
    def search(self: MovieIndex, query: Query) -> list[MovieCandidate]:
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
    async def embed_queries(self: Recommender, queries: list[Query]) -> list[Query]:
        try:
            queries = await self.embedder.to_async.embed(queries)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return queries

    @bentoml.api()
    async def search_movies(self: Recommender, query: Query) -> list[MovieCandidate]:
        try:
            results = await self.movie_index.to_async.search(query)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    async def recommend_with_query(
        self: Recommender, query: Query
    ) -> list[MovieCandidate]:
        try:
            queries = await self.embed_queries([query])
            results = await self.search_movies(queries[0])
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    async def recommend_with_movie(
        self: Recommender, movie: MovieQuery
    ) -> list[MovieCandidate]:
        try:
            results = await self.recommend_with_query(movie.to_query())
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    async def recommend_with_user(
        self: Recommender, user: UserQuery
    ) -> list[MovieCandidate]:
        try:
            results = await self.recommend_with_query(user.to_query())
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results
