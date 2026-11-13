"""Document ingestion pipeline."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from werkzeug.datastructures import FileStorage

from backend.database.repositories import DocumentRepository
from backend.models.entities import Document, DocumentChunk
from backend.services.chunking import chunk_document
from backend.services.embeddings import Embedder
from backend.services.extraction import extract_document
from backend.services.vector_store import VectorStore
from backend.utils.errors import (
    APIError,
    ConflictError,
    EmptyDocumentError,
    PayloadTooLargeError,
    ValidationError,
)
from backend.utils.files import (
    ensure_within_directory,
    get_extension,
    make_stored_filename,
    safe_unlink,
    sanitize_original_filename,
)

logger = logging.getLogger("semanticsearch.ingestion")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IngestionService:
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

    def _save_upload(self, upload: FileStorage) -> tuple[Path, str, int]:
        """Stream an upload to a temporary file while hashing it."""
        temp_name = f".upload-{uuid.uuid4().hex}.tmp"
        temp_path = ensure_within_directory(self.config.upload_dir / temp_name, self.config.upload_dir)

        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("wb") as handle:
                while True:
                    block = upload.stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > self.config.max_content_length:
                        raise PayloadTooLargeError(
                            f"File exceeds the {self.config.max_upload_mb} MB upload limit."
                        )
                    digest.update(block)
                    handle.write(block)
        except PayloadTooLargeError:
            safe_unlink(temp_path, logger)
            raise
        except OSError as exc:
            safe_unlink(temp_path, logger)
            raise APIError(
                "The uploaded file could not be stored.",
                status_code=500,
                code="storage_error",
            ) from exc
        except Exception:
            safe_unlink(temp_path, logger)
            raise

        if size == 0:
            safe_unlink(temp_path, logger)
            raise ValidationError("The uploaded file is empty.")

        return temp_path, digest.hexdigest(), size

    def ingest(self, upload: FileStorage) -> Document:
        original_name = sanitize_original_filename(upload.filename)
        extension = get_extension(original_name)
        if extension not in self.config.allowed_extensions:
            allowed = ", ".join(f".{ext}" for ext in self.config.allowed_extensions)
            raise ValidationError(
                f"Unsupported file type. Allowed extensions: {allowed}.",
                details={"allowed_extensions": list(self.config.allowed_extensions)},
            )

        temp_path, file_hash, file_size = self._save_upload(upload)
        final_path: Path | None = None
        document_id: str | None = None
        document: Document | None = None
        chunk_ids: list[str] = []

        try:
            existing = self.repository.get_document_by_hash(file_hash)
            if existing is not None:
                raise ConflictError(
                    f"'{existing.original_name}' has already been uploaded.",
                    details={"document_id": existing.id, "document": existing.to_dict()},
                )

            # Validate and extract before touching the persistent index.
            extracted = extract_document(temp_path, extension)
            payloads = chunk_document(
                extracted,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            if not payloads:
                raise EmptyDocumentError(
                    "The document produced no text chunks after processing."
                )

            # Fail early if the index was built with a different model.
            self.vector_store.ensure_compatible(self.embedder.name)

            vectors = self.embedder.encode_texts(
                [payload.content for payload in payloads],
                batch_size=self.config.embedding_batch_size,
            )
            if vectors.shape[0] != len(payloads):
                raise APIError(
                    "Embedding generation returned an unexpected number of vectors.",
                    status_code=500,
                    code="embedding_error",
                )

            stored_name = make_stored_filename(extension)
            final_path = ensure_within_directory(
                self.config.upload_dir / stored_name, self.config.upload_dir
            )
            mime_type = "application/pdf" if extension == "pdf" else "text/plain"

            document_id = uuid.uuid4().hex
            created_at = utc_now_iso()
            chunk_ids = [uuid.uuid4().hex for _ in payloads]
            chunks = [
                DocumentChunk(
                    id=chunk_ids[index],
                    document_id=document_id,
                    chunk_index=payload.chunk_index,
                    page_number=payload.page_number,
                    content=payload.content,
                    vector_position=None,
                    created_at=created_at,
                )
                for index, payload in enumerate(payloads)
            ]
            document = Document(
                id=document_id,
                original_name=original_name,
                stored_name=stored_name,
                file_hash=file_hash,
                file_size=file_size,
                mime_type=mime_type,
                page_count=extracted.page_count,
                chunk_count=len(chunks),
                status="ready",
                created_at=created_at,
            )

            with self.vector_store.lock:
                # Re-check inside the lock to close the duplicate-upload race.
                existing = self.repository.get_document_by_hash(file_hash)
                if existing is not None:
                    raise ConflictError(
                        f"'{existing.original_name}' has already been uploaded.",
                        details={
                            "document_id": existing.id,
                            "document": existing.to_dict(),
                        },
                    )

                os.replace(temp_path, final_path)

                try:
                    self.repository.create_document(document, chunks)
                except sqlite3.IntegrityError as exc:
                    existing = self.repository.get_document_by_hash(file_hash)
                    if existing is not None:
                        raise ConflictError(
                            f"'{existing.original_name}' has already been uploaded.",
                            details={
                                "document_id": existing.id,
                                "document": existing.to_dict(),
                            },
                        ) from exc
                    raise APIError(
                        "The document metadata could not be saved.",
                        status_code=500,
                        code="database_error",
                    ) from exc

                try:
                    positions = self.vector_store.add(
                        chunk_ids, vectors, embedding_model=self.embedder.name
                    )
                    self.repository.update_vector_positions(positions)
                except Exception:
                    # Roll the database and file changes back so retries start clean.
                    try:
                        self.vector_store.remove(chunk_ids)
                    except Exception:  # pragma: no cover - defensive cleanup
                        logger.exception(
                            "Could not remove vectors after a failed ingestion; "
                            "search is still protected by the database join."
                        )
                    self.repository.delete_document(document_id)
                    safe_unlink(final_path, logger)
                    final_path = None
                    raise

            logger.info(
                "Ingested '%s' (%s bytes, %s chunks, doc_id=%s)",
                original_name,
                file_size,
                len(chunks),
                document_id,
            )
            return document

        except Exception:
            if final_path is not None:
                safe_unlink(final_path, logger)
            raise
        finally:
            safe_unlink(temp_path, logger)
