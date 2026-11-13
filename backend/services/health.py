"""Health and statistics reporting without loading the embedding model."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from backend.database.repositories import DocumentRepository
from backend.services.embeddings import Embedder
from backend.services.vector_store import VectorStore


class HealthService:
    def __init__(
        self,
        *,
        repository: DocumentRepository,
        embedder: Embedder,
        vector_store: VectorStore,
        version: str,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store
        self.version = version

    def snapshot(self) -> tuple[dict, int]:
        database_ok = self.repository.health_check()
        document_count = 0
        chunk_count = 0
        if database_ok:
            try:
                document_count = self.repository.count_documents()
                chunk_count = self.repository.count_ready_chunks()
            except sqlite3.Error:
                database_ok = False

        vector_count = self.vector_store.count
        model_matches = vector_count == 0 or (
            self.vector_store.embedding_model is not None
            and self.vector_store.embedding_model == self.embedder.name
        )
        index_in_sync = (
            database_ok
            and vector_count == chunk_count
            and model_matches
        )

        warnings: list[str] = []
        if database_ok and vector_count != chunk_count:
            warnings.append(
                "The vector index and database contain different numbers of chunks. "
                "Rebuild the index."
            )
        if database_ok and not model_matches:
            warnings.append(
                "The vector index was built with a different embedding model. "
                "Rebuild the index."
            )
        load_warning = self.vector_store.describe().get("load_warning")
        if load_warning:
            warnings.append(load_warning)

        embedder_info = self.embedder.describe()
        payload = {
            "status": "ok" if database_ok else "error",
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "database": {
                "connected": database_ok,
                "documents": document_count,
                "chunks": chunk_count,
            },
            "index": {
                "in_sync": index_in_sync,
                "vectors": vector_count,
                **{
                    key: value
                    for key, value in self.vector_store.describe().items()
                    if key not in {"count", "load_warning"}
                },
            },
            "embedding": embedder_info,
            "warnings": warnings,
        }
        return payload, (200 if database_ok else 503)
