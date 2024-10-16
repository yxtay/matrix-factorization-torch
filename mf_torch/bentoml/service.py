from __future__ import annotations

from typing import Self

import bentoml
import torch
from loguru import logger

from mf_torch.bentoml.prepare import embed_query, load_args
from mf_torch.bentoml.schemas import ItemCandidate, ItemQuery, Query, UserQuery
from mf_torch.params import (
    CHECKPOINT_PATH,
    EXPORTED_PROGRAM_PATH,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MODEL_NAME,
    USERS_TABLE_NAME,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        path = self.model_ref.path_of(EXPORTED_PROGRAM_PATH)
        self.model = torch.export.load(path).module()
        logger.info("model loaded: {}", path)

    @bentoml.api()
    @logger.catch(reraise=True)
    def embed(self: Self, query: Query) -> Query:
        return embed_query(query=query, model=self.model)


@bentoml.service()
class ItemIndex:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        import lancedb

        src_path = self.model_ref.path_of(LANCE_DB_PATH)
        db = lancedb.connect(src_path)
        self.items = db.open_table(ITEMS_TABLE_NAME)
        self.users = db.open_table(USERS_TABLE_NAME)
        logger.info("items index loaded: {}", src_path)

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(
        self: Self, query: Query, exclude_items: list[int]
    ) -> list[ItemCandidate]:
        from pydantic import TypeAdapter

        exclude_filter = ", ".join(f"{item}" for item in exclude_items)
        exclude_filter = f"movie_id NOT IN ({exclude_filter})"
        results_df = (
            self.items.search(query.embedding)
            .where(exclude_filter, prefilter=True)
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

        result = self.items.search().where(f"movie_id = {item_id}").to_list()
        if len(result) == 0:
            msg = f"item not found: {item_id = }"
            raise NotFound(msg)
        return ItemCandidate.model_validate(result[0])

    @bentoml.api()
    @logger.catch(reraise=True)
    def user_id(self: Self, user_id: int) -> UserQuery:
        from bentoml.exceptions import NotFound

        result = self.users.search().where(f"user_id = {user_id}").to_list()
        if len(result) == 0:
            msg = f"user not found: {user_id = }"
            raise NotFound(msg)
        return UserQuery.model_validate(result[0])


@bentoml.service()
class Service:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)
    embedder = bentoml.depends(Embedder)
    item_index = bentoml.depends(ItemIndex)

    @logger.catch(reraise=True)
    def __init__(self: Self) -> None:
        args = load_args(self.model_ref.path_of(CHECKPOINT_PATH))
        self.num_hashes = args["data"]["num_hashes"]
        self.num_embeddings = args["data"]["num_embeddings"]

    @bentoml.api()
    @logger.catch(reraise=True)
    async def embed_query(self: Self, query: Query) -> Query:
        return await self.embedder.to_async.embed(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def search_items(
        self: Self, query: Query, exclude_items: list[int] | None = None
    ) -> list[ItemCandidate]:
        exclude_items = exclude_items or []
        return await self.item_index.to_async.search(query, exclude_items=exclude_items)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_query(
        self: Self, query: Query, exclude_items: list[int] | None = None
    ) -> list[ItemCandidate]:
        query = await self.embed_query(query)
        return await self.search_items(query, exclude_items=exclude_items)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item(
        self: Self, item: ItemQuery, exclude_items: list[int] | None = None
    ) -> list[ItemCandidate]:
        if item.movie_id:
            exclude_items = [*(exclude_items or []), item.movie_id]

        query = item.to_query(
            num_hashes=self.num_hashes, num_embeddings=self.num_embeddings
        )
        return await self.recommend_with_query(query, exclude_items=exclude_items)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item_id(
        self: Self, item_id: int, exclude_items: list[int] | None = None
    ) -> list[ItemCandidate]:
        item = await self.item_id(item_id)
        return await self.recommend_with_item(item, exclude_items=exclude_items)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def item_id(self: Self, item_id: int) -> ItemCandidate:
        return await self.item_index.to_async.item_id(item_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user(
        self: Self, user: UserQuery, exclude_items: list[int] | None = None
    ) -> list[ItemCandidate]:
        if user.movie_ids:
            exclude_items = exclude_items or []
            exclude_items = [*exclude_items, *user.movie_ids]

        query = user.to_query(
            num_hashes=self.num_hashes, num_embeddings=self.num_embeddings
        )
        return await self.recommend_with_query(query, exclude_items=exclude_items)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user_id(
        self: Self, user_id: int, exclude_items: list[int] | None = None
    ) -> list[ItemCandidate]:
        user = await self.user_id(user_id)
        return await self.recommend_with_user(user, exclude_items=exclude_items)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def user_id(self: Self, user_id: int) -> UserQuery:
        return await self.item_index.to_async.user_id(user_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name
