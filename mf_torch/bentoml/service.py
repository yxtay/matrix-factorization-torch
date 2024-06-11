from __future__ import annotations

import bentoml
import torch
from docarray import DocList
from loguru import logger

from mf_torch.bentoml.models import (
    EMBEDDER_PATH,
    LANCE_DB_PATH,
    MODEL_NAME,
    MOVIES_TABLE_NAME,
    MovieCandidate,
    MovieQuery,
    Query,
    UserQuery,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.get(MODEL_NAME)

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
            with torch.inference_mode():
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
    model_ref = bentoml.models.get(MODEL_NAME)

    def __init__(self: MovieIndex) -> None:
        import lancedb

        src_path = self.model_ref.path_of(LANCE_DB_PATH)
        self.tbl = lancedb.connect(src_path).open_table(MOVIES_TABLE_NAME)
        logger.info("movies index loaded: {}", src_path)
        self.refine_factor = 5

    @bentoml.api()
    def search(self: MovieIndex, query: Query) -> list[MovieCandidate]:
        try:
            results_df = (
                self.tbl.search(query.embedding)
                .refine_factor(self.refine_factor)
                .to_pandas()
                .assign(score=lambda df: 1 - df["_distance"])
                .drop(columns="_distance")
            )
            results = DocList[MovieCandidate].from_dataframe(results_df)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results


@bentoml.service()
class Service:
    model_ref = bentoml.models.get(MODEL_NAME)
    embedder = bentoml.depends(Embedder)
    movie_index = bentoml.depends(MovieIndex)

    @bentoml.api()
    async def embed_queries(self: Service, queries: list[Query]) -> list[Query]:
        try:
            queries = await self.embedder.to_async.embed(queries)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return queries

    @bentoml.api()
    async def search_movies(self: Service, query: Query) -> list[MovieCandidate]:
        try:
            results = await self.movie_index.to_async.search(query)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    async def recommend_with_query(self: Service, query: Query) -> list[MovieCandidate]:
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
        self: Service, movie: MovieQuery
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
        self: Service, user: UserQuery
    ) -> list[MovieCandidate]:
        try:
            results = await self.recommend_with_query(user.to_query())
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name

    @bentoml.api()
    async def version(self: Service) -> str:
        return bentoml.get(MODEL_NAME).tag.version

    @bentoml.api()
    async def name(self: Service) -> str:
        return bentoml.get(MODEL_NAME).tag.name
