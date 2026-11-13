"""Data-access layer for documents and chunks.

All SQL lives here. Services work with domain objects rather than sqlite3
rows, which keeps business logic testable and the database replaceable.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable

from backend.database.connection import connect
from backend.models.entities import Document, DocumentChunk


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        original_name=row["original_name"],
        stored_name=row["stored_name"],
        file_hash=row["file_hash"],
        file_size=row["file_size"],
        mime_type=row["mime_type"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        error_message=row["error_message"],
        created_at=row["created_at"],
    )


def _row_to_chunk(row: sqlite3.Row, document_name: str | None = None) -> DocumentChunk:
    return DocumentChunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        page_number=row["page_number"],
        content=row["content"],
        vector_position=row["vector_position"],
        created_at=row["created_at"],
        document_name=document_name,
    )


class DocumentRepository:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    # ------------------------------------------------------------------ reads
    def get_document(self, document_id: str) -> Document | None:
        with closing(connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> Document | None:
        with closing(connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_document(row) for row in rows]

    def count_documents(self) -> int:
        with closing(connect(self.database_path)) as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def count_chunks(self) -> int:
        with closing(connect(self.database_path)) as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def count_ready_chunks(self) -> int:
        with closing(connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS n
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'ready'
                """
            ).fetchone()
        return int(row["n"])

    def get_chunks_for_document(self, document_id: str) -> list[DocumentChunk]:
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def get_all_chunks(self) -> list[DocumentChunk]:
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT c.*, d.original_name AS document_name
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'ready'
                ORDER BY c.document_id, c.chunk_index
                """
            ).fetchall()
        return [_row_to_chunk(row, row["document_name"]) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[DocumentChunk]:
        """Fetch only chunks whose parent document is still ready.

        This join is a second line of defence: even if a crash leaves a stale
        vector in FAISS, deleted documents can never surface in search results.
        """
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT c.*, d.original_name AS document_name
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders}) AND d.status = 'ready'
        """
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(query, ids).fetchall()
        return [_row_to_chunk(row, row["document_name"]) for row in rows]

    def get_document_ids_for_chunks(self, chunk_ids: Iterable[str]) -> set[str]:
        ids = list(chunk_ids)
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(
                f"SELECT DISTINCT document_id FROM chunks WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row["document_id"] for row in rows}

    # ----------------------------------------------------------------- writes
    def create_document(self, document: Document, chunks: list[DocumentChunk]) -> None:
        with closing(connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, original_name, stored_name, file_hash, file_size,
                        mime_type, page_count, chunk_count, status, error_message,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.original_name,
                        document.stored_name,
                        document.file_hash,
                        document.file_size,
                        document.mime_type,
                        document.page_count,
                        document.chunk_count,
                        document.status,
                        document.error_message,
                        document.created_at,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, page_number, content,
                        vector_position, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.chunk_index,
                            chunk.page_number,
                            chunk.content,
                            chunk.vector_position,
                            chunk.created_at,
                        )
                        for chunk in chunks
                    ],
                )

    def delete_document(self, document_id: str) -> Document | None:
        with closing(connect(self.database_path)) as connection:
            with connection:
                row = connection.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if row is None:
                    return None
                connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return _row_to_document(row)

    def update_vector_positions(self, positions: dict[str, int]) -> None:
        if not positions:
            return
        with closing(connect(self.database_path)) as connection:
            with connection:
                connection.executemany(
                    "UPDATE chunks SET vector_position = ? WHERE id = ?",
                    [(position, chunk_id) for chunk_id, position in positions.items()],
                )

    def clear_vector_positions(self) -> None:
        with closing(connect(self.database_path)) as connection:
            with connection:
                connection.execute("UPDATE chunks SET vector_position = NULL")

    def mark_document_failed(self, document_id: str, message: str) -> None:
        with closing(connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE documents SET status = 'failed', error_message = ? WHERE id = ?",
                    (message[:1000], document_id),
                )

    def health_check(self) -> bool:
        try:
            with closing(connect(self.database_path)) as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
