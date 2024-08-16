from __future__ import annotations

from typing import Self

from docarray.base_doc.doc import BaseDocWithoutId
from docarray.typing import NdArray  # noqa: TCH002

from mf_torch.params import (
    ITEM_FEATURE_NAMES,
    ITEM_IDX,
    USER_FEATURE_NAMES,
    USER_IDX,
)


class Query(BaseDocWithoutId):
    feature_values: list[str]
    feature_hashes: NdArray
    feature_weights: NdArray
    embedding: NdArray | None = None


class ItemQuery(BaseDocWithoutId):
    movie_id: int | None = None
    title: str | None = None
    genres: list[str] | None = None

    def to_query(self: Self, **kwargs: dict[str, int]) -> Query:
        from mf_torch.data.load import process_features

        query_dict = process_features(
            self.model_dump(), idx=ITEM_IDX, feature_names=ITEM_FEATURE_NAMES, **kwargs
        )
        return Query.model_validate(query_dict)


class ItemCandidate(Query, ItemQuery):
    movie_id: int
    title: str
    genres: list[str]
    score: float | None = None


class UserQuery(BaseDocWithoutId):
    user_id: int | None = None
    gender: str | None = None
    age: int | None = None
    occupation: int | None = None
    zipcode: str | None = None

    def to_query(self: Self, **kwargs: dict[str, int]) -> Query:
        from mf_torch.data.load import process_features

        query_dict = process_features(
            self.model_dump(), idx=USER_IDX, feature_names=USER_FEATURE_NAMES, **kwargs
        )
        return Query.model_validate(query_dict)


sample_item_query = ItemQuery(
    movie_id=1,
    title="Toy Story (1995)",
    genres=["Animation", "Children's", "Comedy"],
)
sample_query = sample_item_query.to_query()
sample_user_query = UserQuery(
    user_id=1,
    gender="F",
    age=1,
    occupation=10,
    zipcode="48067",
)
