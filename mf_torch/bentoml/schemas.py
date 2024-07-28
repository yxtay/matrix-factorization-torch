from __future__ import annotations

from typing import Self

from docarray import BaseDoc
from docarray.typing import NdArray  # noqa: TCH002
from lancedb.pydantic import LanceModel, Vector

from mf_torch.params import (
    ITEM_FEATURE_NAMES,
    ITEM_IDX,
    USER_FEATURE_NAMES,
    USER_IDX,
)


class Query(BaseDoc):
    idx: int
    feature_values: list[str]
    feature_hashes: NdArray
    feature_weights: NdArray
    embedding: NdArray[32] | None = None


class MovieQuery(BaseDoc):
    movie_id: int | None = None
    title: str | None = None
    genres: list[str] | None = None

    def to_query(self: Self) -> Query:
        from mf_torch.data.load import process_features

        query_dict = process_features(
            self.model_dump(), idx=ITEM_IDX, feature_names=ITEM_FEATURE_NAMES
        )
        return Query.model_validate(query_dict)


class MovieCandidate(MovieQuery):
    movie_id: int
    title: str
    genres: list[str]
    feature_hashes: NdArray = None
    feature_weights: NdArray = None
    embedding: NdArray[32] = None
    score: float | None = None


class MovieSchema(LanceModel):
    id: str
    movie_id: int
    title: str
    genres: list[str]
    feature_hashes: list[float]
    feature_weights: list[float]
    embedding: Vector(32)


class UserQuery(BaseDoc):
    user_id: int | None = None
    gender: str | None = None
    age: int | None = None
    occupation: int | None = None
    zipcode: str | None = None

    def to_query(self: Self) -> Query:
        from mf_torch.data.load import process_features

        query_dict = process_features(
            self.model_dump(), idx=USER_IDX, feature_names=USER_FEATURE_NAMES
        )
        return Query.model_validate(query_dict)


sample_movie_query = MovieQuery(
    id="1",
    movie_id=1,
    title="Toy Story (1995)",
    genres=["Animation", "Children's", "Comedy"],
)
sample_query = sample_movie_query.to_query()
sample_user_query = UserQuery(
    id="1",
    user_id=1,
    gender="F",
    age=1,
    occupation=10,
    zipcode="48067",
)
