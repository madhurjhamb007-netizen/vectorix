"""Semantic search service."""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.database.repositories import DocumentRepository
from backend.models.entities import SearchResult
from backend.services.embeddings import Embedder
from backend.services.vector_store import VectorStore
from backend.utils.errors import ValidationError

logger = logging.getLogger("semanticsearch.search")


class SearchService:
    def __init__(
        self,
        *,
        config,
        repository: DocumentRepository,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store

    def _validate_query(self, query: Any) -> str:
        if not isinstance(query, str):
            raise ValidationError("The 'query' field must be a string.")
        query = query.strip()
        if not query:
            raise ValidationError("Please enter a search query.")
        if len(query) > self.config.max_query_length:
            raise ValidationError(
                f"Search query must be at most {self.config.max_query_length} characters."
            )
        return query

    def _validate_top_k(self, top_k: Any) -> int:
        if top_k is None:
            return self.config.top_k_default
        if isinstance(top_k, bool):
            raise ValidationError("'top_k' must be an integer.")
        try:
            value = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ValidationError("'top_k' must be an integer.") from exc
        if value < 1 or value > self.config.top_k_max:
            raise ValidationError(
                f"'top_k' must be between 1 and {self.config.top_k_max}."
            )
        return value

    def search(self, query: Any, top_k: Any = None) -> dict:
        started = time.perf_counter()
        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)

        chunk_count = self.repository.count_ready_chunks()
        if chunk_count == 0:
            return self._response(
                query=query,
                top_k=top_k,
                results=[],
                started=started,
                reason="no_documents",
                message="Upload a document to start searching.",
            )

        if self.vector_store.is_empty:
            return self._response(
                query=query,
                top_k=top_k,
                results=[],
                started=started,
                reason="index_empty",
                message=(
                    "The vector index is empty while documents exist. "
                    "Rebuild the index to make these documents searchable."
                ),
            )

        self.vector_store.ensure_compatible(
            self.embedder.name, self.embedder.dimension
        )

        query_vector = self.embedder.encode_query(query)

        # Over-fetch to tolerate vectors that belong to documents deleted
        # between an earlier crash and the last vector cleanup.
        fetch_k = min(max(top_k * 3, top_k + 5), self.vector_store.count)
        candidates = self.vector_store.search(query_vector, fetch_k)
        if not candidates:
            return self._response(
                query=query,
                top_k=top_k,
                results=[],
                started=started,
                reason="no_matches",
                message="No semantically similar passages were found.",
            )

        chunks = self.repository.get_chunks_by_ids(
            [chunk_id for chunk_id, _ in candidates]
        )
        by_id = {chunk.id: chunk for chunk in chunks}

        results: list[SearchResult] = []
        for chunk_id, score in candidates:
            chunk = by_id.get(chunk_id)
            if chunk is None or chunk.document_name is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.content,
                    score=max(-1.0, min(1.0, float(score))),
                )
            )
            if len(results) >= top_k:
                break

        reason = None if results else "no_matches"
        message = None if results else "No semantically similar passages were found."
        return self._response(
            query=query,
            top_k=top_k,
            results=results,
            started=started,
            reason=reason,
            message=message,
        )

    def _response(
        self,
        *,
        query: str,
        top_k: int,
        results: list[SearchResult],
        started: float,
        reason: str | None = None,
        message: str | None = None,
    ) -> dict:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "query": query,
            "top_k": top_k,
            "count": len(results),
            "results": [result.to_dict() for result in results],
            "duration_ms": duration_ms,
            "reason": reason,
            "message": message,
            "index": {
                "backend": self.vector_store.backend_name,
                "vectors": self.vector_store.count,
                "dimension": self.vector_store.dimension,
                "embedding_model": self.vector_store.embedding_model
                or self.embedder.name,
            },
        }
