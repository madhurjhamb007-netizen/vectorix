"""Document listing, deletion, and index rebuilding."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from backend.database.repositories import DocumentRepository
from backend.models.entities import Document
from backend.services.embeddings import Embedder
from backend.services.vector_store import VectorStore
from backend.utils.errors import NotFoundError, ValidationError
from backend.utils.files import ensure_within_directory, safe_unlink

logger = logging.getLogger("semanticsearch.documents")


def validate_document_id(document_id: str) -> str:
    try:
        parsed = uuid.UUID(str(document_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationError("Invalid document ID.") from exc
    return parsed.hex


class DocumentService:
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

    def list_documents(self) -> list[Document]:
        return self.repository.list_documents()

    def get_document(self, document_id: str) -> Document:
        document = self.repository.get_document(validate_document_id(document_id))
        if document is None:
            raise NotFoundError("Document not found.")
        return document

    def delete_document(self, document_id: str) -> dict:
        document_id = validate_document_id(document_id)
        warnings: list[str] = []

        with self.vector_store.lock:
            document = self.repository.get_document(document_id)
            if document is None:
                raise NotFoundError("Document not found.")

            chunks = self.repository.get_chunks_for_document(document_id)
            chunk_ids = [chunk.id for chunk in chunks]

            removed_vectors = self.vector_store.remove(chunk_ids)
            try:
                # Deleted chunks are gone; update every remaining position so
                # SQLite metadata mirrors the compacted FAISS positions.
                self.repository.update_vector_positions(self.vector_store.position_map())
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Could not synchronise vector positions")
                warnings.append("Vector positions could not be synchronised.")

            deleted = self.repository.delete_document(document_id)
            if deleted is None:  # pragma: no cover - guarded by the lock
                raise NotFoundError("Document not found.")

        file_path = ensure_within_directory(
            self.config.upload_dir / document.stored_name, self.config.upload_dir
        )
        if not safe_unlink(file_path, logger):
            warnings.append("The stored file could not be removed from disk.")

        logger.info(
            "Deleted document '%s' (doc_id=%s, chunks=%s, vectors=%s)",
            document.original_name,
            document_id,
            len(chunk_ids),
            removed_vectors,
        )
        return {
            "deleted": True,
            "document": document.to_dict(),
            "chunks_removed": len(chunk_ids),
            "vectors_removed": removed_vectors,
            "warnings": warnings,
        }

    def rebuild_index(self) -> dict:
        """Re-embed every ready chunk and atomically replace the index."""
        started = time.perf_counter()
        with self.vector_store.lock:
            chunks = self.repository.get_all_chunks()
            if not chunks:
                self.vector_store.replace_all(
                    [], np.zeros((0, 0), dtype=np.float32), embedding_model=self.embedder.name
                )
                self.repository.clear_vector_positions()
                return {
                    "documents": self.repository.count_documents(),
                    "chunks": 0,
                    "dimension": 0,
                    "embedding_model": self.embedder.name,
                    "backend": self.vector_store.backend_name,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }

            vectors = self.embedder.encode_texts(
                [chunk.content for chunk in chunks],
                batch_size=self.config.embedding_batch_size,
            )
            if vectors.shape[0] != len(chunks):
                raise RuntimeError("Embedding service returned an unexpected vector count.")

            chunk_ids = [chunk.id for chunk in chunks]
            self.vector_store.replace_all(
                chunk_ids, vectors, embedding_model=self.embedder.name
            )
            self.repository.update_vector_positions(self.vector_store.position_map())

        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Rebuilt vector index: %s chunks, dimension=%s, %sms",
            len(chunks),
            self.vector_store.dimension,
            duration_ms,
        )
        return {
            "documents": self.repository.count_documents(),
            "chunks": len(chunks),
            "dimension": self.vector_store.dimension,
            "embedding_model": self.embedder.name,
            "backend": self.vector_store.backend_name,
            "duration_ms": duration_ms,
        }
