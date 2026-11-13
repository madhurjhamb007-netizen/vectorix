"""Embedding backends.

The production backend is sentence-transformers (all-MiniLM-L6-v2) running on
CPU.  A deterministic hashing backend exists for automated tests and for
constrained environments where native PyTorch DLLs cannot be loaded; it is
explicitly opt-in and is not a semantic model.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from backend.utils.errors import EmbeddingError, ValidationError

logger = logging.getLogger("semanticsearch.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HASH_DIGEST_SIZE = 8


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / norms
    return normalized.astype(np.float32, copy=False)


@runtime_checkable
class Embedder(Protocol):
    name: str
    backend: str

    @property
    def dimension(self) -> int | None: ...

    def encode_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def describe(self) -> dict: ...


class SentenceTransformerEmbedder:
    """Lazily-loaded sentence-transformers model on CPU."""

    backend = "sentence-transformers"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: Path | None = None,
        model_factory=None,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._dimension: int | None = None
        self._model_factory = model_factory

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        if self._model is not None:
            return self._model

        logger.info("Loading embedding model '%s' on %s", self.name, self.device)
        try:
            if self._model_factory is not None:
                model = self._model_factory()
            else:
                from sentence_transformers import SentenceTransformer

                kwargs = {"device": self.device}
                if self.cache_dir is not None:
                    kwargs["cache_folder"] = str(self.cache_dir)
                model = SentenceTransformer(self.name, **kwargs)

            if hasattr(model, "eval"):
                model.eval()
            self._model = model
            get_dimension = getattr(model, "get_sentence_embedding_dimension", None)
            if callable(get_dimension):
                dimension = get_dimension()
                if dimension:
                    self._dimension = int(dimension)
            logger.info(
                "Embedding model ready (dimension=%s)", self._dimension or "unknown"
            )
            return model
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.exception("Failed to load embedding model '%s'", self.name)
            raise EmbeddingError(
                "The embedding model could not be loaded. Check your internet "
                "connection for the first download, or pre-download the model. "
                "Under Windows, an Application Control policy may also block "
                f"PyTorch DLLs. Original error: {exc}"
            ) from exc

    def encode_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> np.ndarray:
        if not texts:
            dimension = self._dimension or 0
            return np.zeros((0, dimension), dtype=np.float32)

        cleaned = [str(text or "").strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValidationError("Cannot embed empty text.")

        model = self._load_model()
        try:
            vectors = model.encode(
                cleaned,
                batch_size=max(1, int(batch_size or self.batch_size)),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(cleaned):
            raise EmbeddingError("The embedding model returned an unexpected shape.")
        if not np.isfinite(matrix).all():
            raise EmbeddingError("The embedding model returned non-finite values.")

        if self._dimension is None:
            self._dimension = int(matrix.shape[1])
        elif matrix.shape[1] != self._dimension:
            raise EmbeddingError("Embedding dimension changed between batches.")

        return normalize_rows(matrix)

    def encode_query(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValidationError("Search query cannot be empty.")
        matrix = self.encode_texts([text], batch_size=1)
        return matrix[0]

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.name,
            "device": self.device,
            "dimension": self.dimension,
            "loaded": self.is_loaded,
        }


class HashingEmbedder:
    """Deterministic feature-hashing embedder for tests/offline fallback.

    It hashes unigrams and bigrams into a fixed vector.  It can retrieve
    documents sharing vocabulary but does **not** understand synonyms.  Use
    the sentence-transformers backend for real semantic search.
    """

    backend = "hashing"

    def __init__(self, dimension: int = 384, name: str = "hashing-fallback") -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = int(dimension)
        self.name = name
        self.batch_size = 64

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        tokens = self._tokens(text)
        if not tokens:
            return vector

        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend(
            (f"{tokens[i]} {tokens[i + 1]}", 0.5) for i in range(len(tokens) - 1)
        )

        for feature, weight in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=_HASH_DIGEST_SIZE
            ).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def encode_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        if any(not str(text or "").strip() for text in texts):
            raise ValidationError("Cannot embed empty text.")
        return np.vstack([self._encode_one(str(text)) for text in texts]).astype(
            np.float32
        )

    def encode_query(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValidationError("Search query cannot be empty.")
        return self._encode_one(text)

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.name,
            "device": "cpu",
            "dimension": self.dimension,
            "loaded": True,
            "note": "Deterministic hashing fallback; not a semantic model.",
        }


def create_embedder(config) -> Embedder:
    backend = config.embedding_backend.lower()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            config.embedding_model,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            cache_dir=config.embedding_cache_dir,
        )
    if backend == "hashing":
        return HashingEmbedder(
            dimension=config.embedding_dimension,
            name="hashing-fallback",
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")
