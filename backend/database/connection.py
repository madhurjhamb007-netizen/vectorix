"""SQLite schema and connection management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    stored_name   TEXT NOT NULL UNIQUE,
    file_hash     TEXT NOT NULL UNIQUE,
    file_size     INTEGER NOT NULL,
    mime_type     TEXT NOT NULL DEFAULT 'application/octet-stream',
    page_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ready'
                  CHECK (status IN ('ready', 'processing', 'failed')),
    error_message TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    page_number     INTEGER,
    content         TEXT NOT NULL,
    vector_position INTEGER,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
"""


def connect(database_path: Path | str) -> sqlite3.Connection:
    """Open a connection with safe defaults and dict-like rows."""
    connection = sqlite3.connect(str(database_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_database(database_path: Path | str) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.executescript(SCHEMA_SQL)
        connection.commit()
    finally:
        connection.close()


def ping(database_path: Path | str) -> bool:
    try:
        connection = connect(database_path)
        try:
            connection.execute("SELECT 1").fetchone()
            return True
        finally:
            connection.close()
    except sqlite3.Error:
        return False
