from __future__ import annotations

from typing import Self

import bentoml
import torch
from loguru import logger

from mf_torch.bentoml.prepare import embed_queries, load_args
from mf_torch.bentoml.schemas import ItemCandidate, ItemQuery, Query, UserQuery
from mf_torch.params import (
    CHECKPOINT_PATH,
    EXPORTED_PROGRAM_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.get(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        path = self.model_ref.path_of(EXPORTED_PROGRAM_PATH)
        self.model = torch.export.load(path).module()
        logger.info("model loaded: {}", path)

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    def embed(self: Self, queries: list[Query]) -> list[Query]:
        from docarray import DocList

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

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(self: Self, query: Query) -> list[ItemCandidate]:
        from pydantic import TypeAdapter

        results_df = (
            self.tbl.search(query.embedding)
            .nprobes(20)
            .refine_factor(5)
            .limit(10)
            .to_pandas()
            .assign(score=lambda df: 1 - df["_distance"])
            .drop(columns="_distance")
        )
        return TypeAdapter(list[ItemCandidate]).validate_python(
            results_df.itertuples(), from_attributes=True
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    def item_id(self: Self, item_id: int) -> ItemCandidate:
        from bentoml.exceptions import NotFound

        result = self.tbl.search().where(f"movie_id = {item_id}").to_list()
        if len(result) == 0:
            msg = f"item not found: {item_id = }"
            raise NotFound(msg)
        return ItemCandidate.model_validate(result[0])


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
    async def recommend_with_item_id(self: Self, item_id: int) -> list[ItemCandidate]:
        item = self.item_index.item_id(item_id)
        return await self.recommend_with_item(item)

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
