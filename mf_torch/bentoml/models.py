from __future__ import annotations

from docarray import BaseDoc
from docarray.typing import NdArray  # noqa: TCH002
from lancedb.pydantic import LanceModel, Vector

EMBEDDER_PATH = "scripted_module.pt"
LANCE_DB_PATH = "lancedb"
MODEL_NAME = "mf-torch"
MOVIES_DOC_PATH = "movies"
MOVIES_TABLE_NAME = "movies"


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
        from mf_torch.data.load import hash_features

        return Query.model_validate(
            hash_features(
                self.model_dump(),
                idx="movie_idx",
                feature_names=["movie_id", "genres"],
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
    id: str
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
        from mf_torch.data.load import hash_features

        return Query.model_validate(
            hash_features(
                self.model_dump(),
                idx="user_idx",
                feature_names=["user_id", "gendeer", "age", "occupation", "zipcode"],
                keep_input=False,
            )
        )


sample_movie_query = MovieQuery(
    id="1",
    movie_idx=1,
    movie_id=1,
    title="Toy Story (1995)",
    genres=["Animation", "Children's", "Comedy"],
)
sample_query = sample_movie_query.to_query()
sample_user_query = UserQuery(
    id="1", user_idx=1, user_id=1, gender="F", age=1, occupation=10, zipcode="48067"
)
