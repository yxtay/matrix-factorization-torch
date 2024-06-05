from __future__ import annotations

import bentoml
import bentoml.models
import torch
from docarray import DocList
from docarray.index import InMemoryExactNNIndex
from loguru import logger

from mf_torch.bentoml.models import (
    EMBEDDER_PATH,
    MODEL_TAG,
    MOVIES_DOC_PATH,
    MovieCandidate,
    MovieQuery,
    Query,
    UserQuery,
)


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.get(MODEL_TAG)

    def __init__(self: Embedder) -> None:
        try:
            path = self.model_ref.path_of(EMBEDDER_PATH)
            self.model = torch.jit.load(path)
            logger.info("embedder loaded: {}", path)
        except Exception as e:
            logger.exception(e)
            raise

    @bentoml.api(batchable=True)
    def embed(self: Embedder, queries: list[Query]) -> list[Query]:
        try:
            queries = DocList[Query](queries)
            feature_hashes = torch.nested.nested_tensor(
                queries.feature_hashes
            ).to_padded_tensor(padding=0)
            feature_weights = torch.nested.nested_tensor(
                queries.feature_weights
            ).to_padded_tensor(padding=0)

            queries.embedding = list(self.model(feature_hashes, feature_weights))
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return queries


@bentoml.service()
class DocIndex:
    model_ref = bentoml.models.get(MODEL_TAG)

    def __init__(self: DocIndex) -> None:
        from pathlib import Path

        path = Path(self.model_ref.path_of(MOVIES_DOC_PATH)).as_uri()
        doc_list = DocList[MovieCandidate].pull(path)
        logger.info("documents loaded: {}", path)
        self.doc_index = InMemoryExactNNIndex[MovieCandidate](doc_list)

    @bentoml.api(batchable=True)
    def find(self: DocIndex, queries: list[Query]) -> list[list[MovieCandidate]]:
        try:
            queries = DocList[Query](queries)
            matches, scores = self.doc_index.find_batched(
                torch.as_tensor(queries.embedding), search_field="embedding"
            )
            logger.debug(matches[0][0])
            for i, score in enumerate(scores):
                matches[i].score = score
            logger.debug(matches[0][0])
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return matches


@bentoml.service()
class Recommender:
    embedder = bentoml.depends(Embedder)
    doc_index = bentoml.depends(DocIndex)

    @bentoml.api(batchable=True)
    def recommend(
        self: Recommender, queries: list[Query]
    ) -> list[list[MovieCandidate]]:
        try:
            queries = self.embedder.embed(queries)
            results = self.doc_index.find(queries)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    def recommend_with_movie(
        self: Recommender, movie: MovieQuery
    ) -> list[MovieCandidate]:
        try:
            results = self.recommend([movie.to_query()])[0]
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results

    @bentoml.api()
    def recommend_with_user(self: Recommender, user: UserQuery) -> list[MovieCandidate]:
        try:
            results = self.recommend([user.to_query()])[0]
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return results
