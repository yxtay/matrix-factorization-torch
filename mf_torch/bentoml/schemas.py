from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from numpydantic import NDArray, Shape


class Query(BaseModel):
    feature_values: list[str]
    feature_hashes: NDArray[Shape["*"], int]
    feature_weights: NDArray[Shape["*"], float]
    embedding: NDArray[Shape["*"], float] | None = None


class ItemQuery(BaseModel):
    movie_id: int | None = None
    title: str | None = None
    genres: list[str] | None = None


class ItemCandidate(Query, ItemQuery):
    movie_id: int
    title: str
    genres: list[str]
    score: float | None = None


class UserQuery(BaseModel):
    user_id: int | None = None
    gender: str | None = None
    age: int | None = None
    occupation: int | None = None
    zipcode: str | None = None
    movie_ids: list[int] | None = None
