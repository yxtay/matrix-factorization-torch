from __future__ import annotations

from docarray import BaseDoc
from docarray.typing import NdArray  # noqa: TCH002

from lancedb.pydantic import LanceModel, Vector
from mf_torch.data.lightning import Movielens1mPipeDataModule
from mf_torch.data.load import hash_features

MODEL_NAME = "mf-torch"
MODEL_TAG = f"{MODEL_NAME}:latest"
EMBEDDER_PATH = "scripted_module.pt"
MOVIES_DOC_PATH = "movies"
LANCE_DB_PATH = "lancedb"
LANCE_TABLE_NAME = "movies"


class Query(BaseDoc):
    feature_hashes: NdArray
    feature_weights: NdArray
    embedding: NdArray[32] | None = None


class MovieQuery(BaseDoc):
    movie_idx: int | None = None
    movie_id: int | None = None
    title: str | None = None
    genres: list[str] | None = None

    def to_query(self: MovieQuery) -> Query:
        return Query.model_validate(
            hash_features(
                self.model_dump(),
                idx=Movielens1mPipeDataModule.item_idx,
                feature_names=Movielens1mPipeDataModule.item_feature_names,
                keep_input=False,
            )
        )


class MovieCandidate(MovieQuery):
    movie_idx: int
    movie_id: int
    title: str
    genres: list[str]
    feature_hashes: NdArray = None
    feature_weights: NdArray = None
    embedding: NdArray[32] = None
    score: float | None = None


class MovieSchema(LanceModel):
    movie_idx: int
    movie_id: int
    title: str
    genres: list[str]
    feature_hashes: list[float]
    feature_weights: list[float]
    embedding: Vector(32)


class UserQuery(BaseDoc):
    user_idx: int | None = None
    user_id: int | None = None
    gender: str | None = None
    age: int | None = None
    occupation: int | None = None
    zipcode: str | None = None

    def to_query(self: UserQuery) -> Query:
        return Query.model_validate(
            hash_features(
                self.model_dump(),
                idx=Movielens1mPipeDataModule.user_idx,
                feature_names=Movielens1mPipeDataModule.user_feature_names,
                keep_input=False,
            )
        )
