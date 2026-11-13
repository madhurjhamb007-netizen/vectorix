"""Tests for embedding backends and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    normalize_rows,
)
from backend.utils.errors import ValidationError
from config import AppConfig


class FakeSentenceTransformer:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or [[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]]
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray(self.responses[: len(texts)], dtype=np.float32)


def test_normalize_rows_handles_zero_vectors():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    result = normalize_rows(vectors)
    assert np.allclose(result[0], [0.6, 0.8])
    assert np.allclose(result[1], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(result, axis=1), [1.0, 0.0], atol=1e-6)


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dimension=64, name="hash-test")
    texts = ["supervised learning with labels", "FAISS vector retrieval"]

    first = embedder.encode_texts(texts)
    second = embedder.encode_texts(texts)

    assert first.shape == (2, 64)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert not np.allclose(first[0], first[1])


def test_hashing_embedder_empty_batch_shape():
    embedder = HashingEmbedder(dimension=32)
    result = embedder.encode_texts([])
    assert result.shape == (0, 32)


def test_hashing_embedder_rejects_empty_text():
    with pytest.raises(ValidationError):
        HashingEmbedder(dimension=16).encode_texts(["valid", "  "])


def test_sentence_transformer_embedder_uses_cpu_and_normalizes():
    fake = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        "fake-model", device="cpu", batch_size=2, model_factory=lambda: fake
    )

    vectors = embedder.encode_texts(["first", "second"], batch_size=2)

    assert fake.eval_called is True
    assert embedder.dimension == 2
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    _, kwargs = fake.calls[0]
    assert kwargs["batch_size"] == 2
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["normalize_embeddings"] is False
    assert kwargs["show_progress_bar"] is False


def test_sentence_transformer_query_embedding_and_validation():
    fake = FakeSentenceTransformer(responses=[[3.0, 4.0]])
    embedder = SentenceTransformerEmbedder("fake-model", model_factory=lambda: fake)

    query = embedder.encode_query("question")

    assert np.allclose(query, [0.6, 0.8])
    with pytest.raises(ValidationError):
        embedder.encode_query("   ")


def test_sentence_transformer_rejects_empty_batch_text():
    embedder = SentenceTransformerEmbedder(
        "fake-model", model_factory=lambda: FakeSentenceTransformer()
    )
    with pytest.raises(ValidationError):
        embedder.encode_texts(["valid", ""])


def test_sentence_transformer_zero_vector_stays_zero():
    fake = FakeSentenceTransformer(responses=[[0.0, 0.0]])
    embedder = SentenceTransformerEmbedder("fake-model", model_factory=lambda: fake)
    vectors = embedder.encode_texts(["zero"])
    assert np.allclose(vectors, [[0.0, 0.0]])


def test_create_embedder_honours_hashing_backend(tmp_path):
    config = AppConfig.for_testing(tmp_path)
    embedder = create_embedder(config)
    assert embedder.backend == "hashing"
    assert embedder.name == "hashing-fallback"
    assert embedder.dimension == 32
    assert embedder.describe()["note"].startswith("Deterministic")
