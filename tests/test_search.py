"""Tests for the semantic search service and HTTP endpoint."""

from __future__ import annotations

import numpy as np

from tests.support import upload_pdf, upload_txt


def data(response):
    payload = response.get_json()
    assert payload is not None
    return payload["data"]


def test_search_on_empty_library_is_graceful(client):
    response = client.post("/api/search", json={"query": "what is machine learning?"})

    assert response.status_code == 200
    result = data(response)
    assert result["count"] == 0
    assert result["results"] == []
    assert result["reason"] == "no_documents"
    assert "upload" in result["message"].lower()


def test_search_query_validation(client):
    empty = client.post("/api/search", json={"query": "   "})
    assert empty.status_code == 400
    assert empty.get_json()["error"]["code"] == "validation_error"

    missing = client.post("/api/search", json={"top_k": 3})
    assert missing.status_code == 400

    not_string = client.post("/api/search", json={"query": 123})
    assert not_string.status_code == 400

    not_json = client.post("/api/search", data="query=test")
    assert not_json.status_code == 400

    too_long = client.post("/api/search", json={"query": "x" * 501})
    assert too_long.status_code == 400


def test_top_k_validation(client):
    upload_txt(client, "Some content about documents.")

    for invalid in (0, -2, 9, "abc", True):
        response = client.post("/api/search", json={"query": "documents", "top_k": invalid})
        assert response.status_code == 400, invalid


def test_top_k_is_honoured(client):
    for index in range(5):
        upload_txt(
            client,
            f"Document number {index} about file uploads and text chunks.",
            f"doc-{index}.txt",
        )

    response = client.post(
        "/api/search", json={"query": "document uploads", "top_k": 2}
    )

    assert response.status_code == 200
    result = data(response)
    assert result["top_k"] == 2
    assert 1 <= result["count"] <= 2
    assert len(result["results"]) == result["count"]


def test_pdf_search_returns_correct_page_number(client):
    upload_pdf(
        client,
        [
            "Supervised learning uses labeled examples for classification and regression.",
            "Unsupervised learning discovers groups in unlabeled data through clustering.",
        ],
        filename="learning.pdf",
    )

    response = client.post("/api/search", json={"query": "clustering groups"})

    assert response.status_code == 200
    result = data(response)
    assert result["count"] >= 1
    top = result["results"][0]
    assert top["document_name"] == "learning.pdf"
    assert top["page_number"] == 2
    assert "clustering" in top["text"].lower()
    assert 0 < top["score"] <= 1
    assert "chunk_id" in top and "document_id" in top


def test_meaning_based_retrieval_without_exact_words(client):
    upload_txt(
        client,
        "A deep neural network stacks many learned layers. Training adjusts weights.",
        "neural-networks.txt",
    )
    upload_txt(
        client,
        "Invoices must be paid within thirty days. Contact accounts for questions.",
        "billing.txt",
    )

    response = client.post(
        "/api/search",
        json={"query": "models with many hidden levels", "top_k": 3},
    )

    assert response.status_code == 200
    result = data(response)
    assert result["count"] >= 1
    assert result["results"][0]["document_name"] == "neural-networks.txt"
    assert result["results"][0]["score"] > 0
    # The query does not literally contain "neural" or "layers".
    assert "neural" not in result["query"].lower()
    assert "layers" not in result["query"].lower()


def test_result_scores_and_metadata_are_serializable(client):
    upload_txt(client, "FAISS retrieves nearest vectors by cosine similarity.")
    response = client.post("/api/search", json={"query": "nearest vector retrieval"})
    result = data(response)
    for item in result["results"]:
        assert isinstance(item["score"], float)
        assert -1.0 <= item["score"] <= 1.0
        assert isinstance(item["text"], str) and item["text"]
        assert isinstance(item["page_number"], (int, type(None)))
        assert isinstance(item["chunk_index"], int)


def test_query_embedding_dimension_must_match_index(client, services, embedder):
    upload_txt(client, "A document about text chunking.")
    query = np.ones(embedder.dimension + 1, dtype=np.float32)
    # Call the store directly to verify a corrupt/incompatible query is caught.
    import pytest
    from backend.utils.errors import VectorStoreError

    with pytest.raises(VectorStoreError):
        services.vector_store.search(query, 1)
