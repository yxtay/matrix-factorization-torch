from __future__ import annotations

from typing import Self

import bentoml
import torch
from docarray import DocList
from loguru import logger

from mf_torch.bentoml.prepare import embed_queries, load_args
from mf_torch.bentoml.schemas import ItemCandidate, ItemQuery, Query, UserQuery
from mf_torch.params import (
    CHECKPOINT_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
    SCRIPTMODULE_PATH,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.get(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        path = self.model_ref.path_of(SCRIPTMODULE_PATH)
        self.model = torch.jit.load(path)
        logger.info("model loaded: {}", path)

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    def embed(self: Self, queries: list[Query]) -> list[Query]:
        queries = DocList[Query](queries)
        return embed_queries(queries=queries, model=self.model)


@bentoml.service()
class ItemIndex:
    model_ref = bentoml.models.get(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        import lancedb

        src_path = self.model_ref.path_of(LANCE_DB_PATH)
        self.tbl = lancedb.connect(src_path).open_table(ITEMS_TABLE_NAME)
        logger.info("items index loaded: {}", src_path)
        self.refine_factor = 5

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(self: Self, query: Query) -> list[ItemCandidate]:
        results_df = (
            self.tbl.search(query.embedding)
            .refine_factor(self.refine_factor)
            .to_pandas()
            .assign(score=lambda df: 1 - df["_distance"])
            .drop(columns="_distance")
        )
        return DocList[ItemCandidate].from_dataframe(results_df)


@bentoml.service()
class Service:
    model_ref = bentoml.models.get(MODEL_NAME)
    embedder = bentoml.depends(Embedder)
    item_index = bentoml.depends(ItemIndex)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        args = load_args(self.model_ref.path_of(CHECKPOINT_PATH))
        self.num_hashes = args["data"]["num_hashes"]
        self.num_embeddings = args["data"]["num_embeddings"]

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    async def embed_queries(self: Self, queries: list[Query]) -> list[Query]:
        return await self.embedder.to_async.embed(queries)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def search_items(self: Self, query: Query) -> list[ItemCandidate]:
        return await self.item_index.to_async.search(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_query(self: Self, query: Query) -> list[ItemCandidate]:
        queries = await self.embed_queries([query])
        return await self.search_items(queries[0])

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item(self: Self, item: ItemQuery) -> list[ItemCandidate]:
        query = item.to_query(
            num_hashes=self.num_hashes, num_embeddings=self.num_embeddings
        )
        return await self.recommend_with_query(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user(self: Self, user: UserQuery) -> list[ItemCandidate]:
        query = user.to_query(
            num_hashes=self.num_hashes, num_embeddings=self.num_embeddings
        )
        return await self.recommend_with_query(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name
