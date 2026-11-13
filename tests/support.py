"""Shared deterministic test helpers (no model downloads required)."""

from __future__ import annotations

import io
import re

import numpy as np

from backend.services.embeddings import normalize_rows
from backend.utils.errors import ValidationError

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A hand-built "semantic" space for tests. Words that belong to the same
# concept map to the same dimension even when they are not synonyms in the
# source document. This lets tests verify meaning-based retrieval without
# pulling down a real transformer model.
TOPIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("supervised", "classification", "regression", "label", "labeled", "labels", "prediction", "predictions", "learning"),
    ("unsupervised", "clustering", "cluster", "unlabeled", "structure", "grouping"),
    ("neural", "network", "networks", "deep", "layer", "layers", "architecture", "architectures", "hidden", "levels", "models"),
    ("vector", "faiss", "embedding", "embeddings", "similarity", "nearest", "neighbours", "neighbors", "index", "vectors"),
    ("overfitting", "overfit", "regularization", "generalization", "generalize", "memorize", "noise"),
    ("search", "semantic", "retrieval", "query", "queries", "meaning", "relevant", "passages"),
    ("python", "flask", "backend", "api", "server", "endpoint", "sqlite"),
    ("document", "documents", "upload", "chunk", "chunks", "pdf", "text", "file", "files"),
)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "what", "which", "how", "why", "is", "are", "was", "were", "does", "do",
    "can", "could", "should", "would", "many", "much", "from", "by", "as",
    "that", "this", "these", "those", "it", "its", "be", "been", "being",
}


class TopicEmbedder:
    """Deterministic bag-of-concepts embedder used by the test suite."""

    backend = "test"
    name = "test-topic-embedder"
    batch_size = 8

    @property
    def dimension(self) -> int:
        return len(TOPIC_GROUPS) + 8

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]
        for token in tokens:
            for index, group in enumerate(TOPIC_GROUPS):
                if token in group:
                    vector[index] += 1.0
                    break
            else:
                # A small stable hash bucket for out-of-vocabulary words so
                # identical rare words still contribute to similarity.
                digest = 0
                for char in token:
                    digest = (digest * 131 + ord(char)) % 8
                vector[len(TOPIC_GROUPS) + digest] += 0.15
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode_texts(self, texts, batch_size=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if any(not str(text or "").strip() for text in texts):
            raise ValidationError("Cannot embed empty text.")
        return normalize_rows(np.vstack([self._encode_one(str(text)) for text in texts]))

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
            "note": "Deterministic test embedder.",
        }


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Create a real, multi-page PDF with extractable text."""
    pymupdf = pytest_import_pymupdf()
    document = pymupdf.open()
    try:
        for text in pages:
            page = document.new_page()
            page.insert_textbox(
                pymupdf.Rect(54, 54, 540, 780),
                text,
                fontsize=11,
                fontname="helv",
            )
        return document.tobytes()
    finally:
        document.close()


def pytest_import_pymupdf():
    try:
        import pymupdf

        return pymupdf
    except ImportError:  # pragma: no cover
        import fitz

        return fitz


def upload_txt(client, content: str | bytes, filename: str = "notes.txt", **kwargs):
    data = content.encode("utf-8") if isinstance(content, str) else content
    return client.post(
        "/api/documents/upload",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
        **kwargs,
    )


def upload_pdf(client, pages: list[str], filename: str = "paper.pdf"):
    return client.post(
        "/api/documents/upload",
        data={"file": (io.BytesIO(make_pdf_bytes(pages)), filename)},
        content_type="multipart/form-data",
    )


def upload_bytes(client, content: bytes, filename: str):
    return client.post(
        "/api/documents/upload",
        data={"file": (io.BytesIO(content), filename)},
        content_type="multipart/form-data",
    )
