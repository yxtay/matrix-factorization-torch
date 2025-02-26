from __future__ import annotations

from typing import Annotated

import bentoml
import pydantic
import torch  # noqa: TC002
from bentoml.validators import DType


class Activity(pydantic.BaseModel):
    movie_id: list[int]
    rating: list[int]


class UserQuery(pydantic.BaseModel):
    user_id: int | None = None
    gender: str | None = None
    age: int | None = None
    occupation: int | None = None
    zipcode: str | None = None
    history: Activity | None = None
    target: Activity | None = None


class ItemQuery(pydantic.BaseModel):
    movie_id: int | None = None
    title: str | None = None
    genres: list[str] | None = None


class Query(bentoml.IODescriptor):
    feature_values: list[str]
    feature_hashes: Annotated[torch.Tensor, DType("int64")]
    feature_weights: Annotated[torch.Tensor, DType("float32")]
    embedding: Annotated[torch.Tensor, DType("float32")] | None = None


class ItemCandidate(pydantic.BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float


EXAMPLE_ITEM = ItemQuery(
    movie_id=1,
    title="Toy Story (1995)",
    genres=["Animation", "Children's", "Comedy"],
)

EXAMPLE_USER = UserQuery(
    user_id=1,
    gender="F",
    age=1,
    occupation=10,
    zipcode="48067",
)
