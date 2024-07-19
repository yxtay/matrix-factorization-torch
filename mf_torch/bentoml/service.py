from __future__ import annotations

from typing import Self

import bentoml
import torch
from docarray import DocList
from loguru import logger

from mf_torch.bentoml.schemas import MovieCandidate, MovieQuery, Query, UserQuery
from mf_torch.params import (
    EMBEDDER_PATH,
    LANCE_DB_PATH,
    MODEL_NAME,
    MOVIES_TABLE_NAME,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.get(MODEL_NAME)

    @logger.catch()
    def __init__(self: Self) -> None:
        path = self.model_ref.path_of(EMBEDDER_PATH)
        self.model = torch.jit.load(path)
        logger.info("embedder loaded: {}", path)

    @bentoml.api(batchable=True)
    @logger.catch()
    def embed(self: Self, queries: list[Query]) -> list[Query]:
        with torch.inference_mode():
            queries = DocList[Query](queries)
            feature_hashes = torch.nested.nested_tensor(
                queries.feature_hashes
            ).to_padded_tensor(padding=0)
            feature_weights = torch.nested.nested_tensor(
                queries.feature_weights
            ).to_padded_tensor(padding=0)

            queries.embedding = list(self.model(feature_hashes, feature_weights))
            return queries


@bentoml.service()
class MovieIndex:
    model_ref = bentoml.models.get(MODEL_NAME)

    @logger.catch()
    def __init__(self: Self) -> None:
        import lancedb

        src_path = self.model_ref.path_of(LANCE_DB_PATH)
        self.tbl = lancedb.connect(src_path).open_table(MOVIES_TABLE_NAME)
        logger.info("movies index loaded: {}", src_path)
        self.refine_factor = 5

    @bentoml.api()
    @logger.catch()
    def search(self: Self, query: Query) -> list[MovieCandidate]:
        results_df = (
            self.tbl.search(query.embedding)
            .refine_factor(self.refine_factor)
            .to_pandas()
            .assign(score=lambda df: 1 - df["_distance"])
            .drop(columns="_distance")
        )
        return DocList[MovieCandidate].from_dataframe(results_df)


@bentoml.service()
class Service:
    model_ref = bentoml.models.get(MODEL_NAME)
    embedder = bentoml.depends(Embedder)
    movie_index = bentoml.depends(MovieIndex)

    @bentoml.api(batchable=True)
    @logger.catch()
    async def embed_queries(self: Self, queries: list[Query]) -> list[Query]:
        return await self.embedder.to_async.embed(queries)

    @bentoml.api()
    @logger.catch()
    async def search_movies(self: Self, query: Query) -> list[MovieCandidate]:
        return await self.movie_index.to_async.search(query)

    @bentoml.api()
    @logger.catch()
    async def recommend_with_query(self: Self, query: Query) -> list[MovieCandidate]:
        queries = await self.embed_queries([query])
        return await self.search_movies(queries[0])

    @bentoml.api()
    @logger.catch()
    async def recommend_with_movie(
        self: Self, movie: MovieQuery
    ) -> list[MovieCandidate]:
        return await self.recommend_with_query(movie.to_query())

    @bentoml.api()
    @logger.catch()
    async def recommend_with_user(self: Self, user: UserQuery) -> list[MovieCandidate]:
        return await self.recommend_with_query(user.to_query())

    @bentoml.api()
    @logger.catch()
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch()
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name
