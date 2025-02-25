from __future__ import annotations

import pydantic
from numpydantic import NDArray, Shape  # noqa: TC002


class Query(pydantic.BaseModel):
    feature_values: list[str]
    feature_hashes: NDArray[Shape["*"], int]  # noqa: F722
    feature_weights: NDArray[Shape["*"], float]  # noqa: F722
    embedding: NDArray[Shape["*"], float] | None = None  # noqa: F722


class ItemQuery(pydantic.BaseModel):
    movie_id: int | None = None
    title: str | None = None
    genres: list[str] | None = None


class ItemCandidate(Query, ItemQuery):
    movie_id: int
    title: str
    genres: list[str]
    score: float | None = None


class UserQuery(pydantic.BaseModel):
    user_id: int | None = None
    gender: str | None = None
    age: int | None = None
    occupation: int | None = None
    zipcode: str | None = None
    movie_ids: list[int] | None = None
