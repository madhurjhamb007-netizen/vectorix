"""Application configuration.

All settings are read from environment variables (optionally via a local
``.env`` file).  No machine-specific paths are hardcoded: relative paths are
resolved against the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_path(name: str, default: Path, base: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_extensions(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    extensions = tuple(
        ext.lower().lstrip(".") for ext in raw.replace(";", ",").split(",") if ext.strip()
    )
    if not extensions:
        raise ValueError(f"{name} must contain at least one file extension")
    return extensions


@dataclass
class AppConfig:
    """Typed configuration object used by the application factory."""

    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    storage_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage")
    upload_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "uploads")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "indexes")
    database_path: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "semantic_search.db")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "logs")

    secret_key: str = "dev-only-change-me"
    log_level: str = "INFO"
    log_to_file: bool = True
    max_content_length: int = 25 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = ("pdf", "txt")

    chunk_size: int = 700
    chunk_overlap: int = 120

    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384
    embedding_cache_dir: Path | None = None

    vector_backend: str = "auto"  # auto | faiss | numpy

    top_k_default: int = 5
    top_k_max: int = 20
    max_query_length: int = 1000

    @classmethod
    def from_env(cls) -> "AppConfig":
        base = PROJECT_ROOT
        storage = _env_path("STORAGE_DIR", base / "storage", base)

        backend = _env_str("EMBEDDING_BACKEND", "sentence-transformers").lower()
        if backend not in {"sentence-transformers", "hashing"}:
            raise ValueError(
                "EMBEDDING_BACKEND must be 'sentence-transformers' or 'hashing'"
            )

        vector_backend = _env_str("VECTOR_BACKEND", "auto").lower()
        if vector_backend not in {"auto", "faiss", "numpy"}:
            raise ValueError("VECTOR_BACKEND must be 'auto', 'faiss' or 'numpy'")

        config = cls(
            base_dir=base,
            storage_dir=storage,
            upload_dir=_env_path("UPLOAD_DIR", storage / "uploads", base),
            index_dir=_env_path("INDEX_DIR", storage / "indexes", base),
            database_path=_env_path("DATABASE_PATH", storage / "semantic_search.db", base),
            log_dir=_env_path("LOG_DIR", storage / "logs", base),
            secret_key=_env_str("SECRET_KEY", "dev-only-change-me"),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
            log_to_file=_env_bool("LOG_TO_FILE", True),
            max_content_length=_env_int("MAX_UPLOAD_MB", 25, minimum=1) * 1024 * 1024,
            allowed_extensions=_env_extensions("ALLOWED_EXTENSIONS", "pdf,txt"),
            chunk_size=_env_int("CHUNK_SIZE", 700, minimum=100),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120, minimum=0),
            embedding_backend=backend,
            embedding_model=_env_str(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_device=_env_str("EMBEDDING_DEVICE", "cpu"),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32, minimum=1),
            embedding_dimension=_env_int("EMBEDDING_DIMENSION", 384, minimum=8),
            embedding_cache_dir=_env_path(
                "EMBEDDING_CACHE_DIR", storage / "models", base
            ),
            vector_backend=vector_backend,
            top_k_default=_env_int("TOP_K_DEFAULT", 5, minimum=1),
            top_k_max=_env_int("TOP_K_MAX", 20, minimum=1),
            max_query_length=_env_int("MAX_QUERY_LENGTH", 1000, minimum=1),
        )
        config.validate()
        return config

    @classmethod
    def for_testing(cls, temp_dir: Path, **overrides: object) -> "AppConfig":
        """Build an isolated configuration suitable for pytest."""
        storage = Path(temp_dir) / "storage"
        config = cls(
            base_dir=temp_dir,
            storage_dir=storage,
            upload_dir=storage / "uploads",
            index_dir=storage / "indexes",
            database_path=storage / "semantic_search.db",
            log_dir=storage / "logs",
            secret_key="test-secret",
            log_level="WARNING",
            log_to_file=False,
            max_content_length=1024 * 1024,
            chunk_size=240,
            chunk_overlap=50,
            embedding_backend="hashing",
            embedding_model="test-hash-embedder",
            embedding_batch_size=4,
            embedding_dimension=32,
            embedding_cache_dir=storage / "models",
            vector_backend="numpy",
            top_k_default=3,
            top_k_max=8,
            max_query_length=500,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        config.validate()
        return config

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k_default > self.top_k_max:
            raise ValueError("TOP_K_DEFAULT cannot exceed TOP_K_MAX")
        if not self.allowed_extensions:
            raise ValueError("At least one allowed file extension is required")

    @property
    def max_upload_mb(self) -> int:
        return max(1, self.max_content_length // (1024 * 1024))

    def ensure_directories(self) -> None:
        for path in (
            self.storage_dir,
            self.upload_dir,
            self.index_dir,
            self.log_dir,
            self.database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.embedding_cache_dir is not None:
            self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
