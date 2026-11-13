"""Tests for index diagnostics, rebuild, and persistence across restarts."""

from __future__ import annotations

import numpy as np

from backend.app_factory import create_app
from tests.support import upload_pdf, upload_txt


def data(response):
    return response.get_json()["data"]


def test_rebuild_recovers_an_empty_index(client, services, embedder):
    upload_txt(
        client,
        "Supervised learning uses labeled examples for classification.",
        "ml.txt",
    )
    upload_pdf(
        client,
        ["Vector search uses cosine similarity.", "FAISS stores dense vectors."],
        "vectors.pdf",
    )
    total_chunks = services.repository.count_ready_chunks()
    assert total_chunks >= 2

    services.vector_store.replace_all(
        [], np.zeros((0, 0), dtype=np.float32), embedding_model=embedder.name
    )

    empty_search = client.post("/api/search", json={"query": "classification"}).get_json()["data"]
    assert empty_search["reason"] == "index_empty"
    assert empty_search["results"] == []

    health = client.get("/api/health").get_json()["data"]
    assert health["index"]["in_sync"] is False

    rebuild = client.post("/api/index/rebuild")
    assert rebuild.status_code == 200
    result = data(rebuild)
    assert result["chunks"] == total_chunks
    assert result["dimension"] == embedder.dimension
    assert result["embedding_model"] == embedder.name

    search = client.post("/api/search", json={"query": "classification labels"}).get_json()["data"]
    assert search["count"] >= 1
    assert search["results"][0]["document_name"] == "ml.txt"

    health = client.get("/api/health").get_json()["data"]
    assert health["index"]["in_sync"] is True
    assert health["database"]["chunks"] == health["index"]["vectors"]

    chunks = services.repository.get_all_chunks()
    assert all(chunk.vector_position is not None for chunk in chunks)


def test_model_mismatch_blocks_search_until_rebuild(client, services, embedder):
    upload_txt(
        client,
        "Neural networks stack layers of learned representations.",
        "neural.txt",
    )
    chunks = services.repository.get_all_chunks()
    vectors = embedder.encode_texts([chunk.content for chunk in chunks])
    services.vector_store.replace_all(
        [chunk.id for chunk in chunks],
        vectors,
        embedding_model="an-old-embedding-model",
    )

    response = client.post("/api/search", json={"query": "neural layers"})
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "index_incompatible"

    health = client.get("/api/health").get_json()["data"]
    assert health["index"]["in_sync"] is False

    rebuild = client.post("/api/index/rebuild")
    assert rebuild.status_code == 200

    response = client.post("/api/search", json={"query": "neural layers"})
    assert response.status_code == 200
    assert response.get_json()["data"]["count"] >= 1


def test_rebuild_empty_library_is_safe(client):
    response = client.post("/api/index/rebuild")
    assert response.status_code == 200
    result = data(response)
    assert result["chunks"] == 0
    assert result["documents"] == 0

    health = client.get("/api/health").get_json()["data"]
    assert health["index"]["in_sync"] is True


def test_index_and_metadata_survive_application_restart(
    test_config, embedder, client, services
):
    document = data(
        upload_txt(
            client,
            "Persistent semantic search stores vectors on disk between restarts.",
            "persistent.txt",
        )
    )
    chunk_count = document["chunk_count"]

    reloaded_app = create_app(test_config, embedder=embedder)
    reloaded_app.config.update(TESTING=True)

    assert reloaded_app.extensions["services"].vector_store.count == chunk_count

    with reloaded_app.test_client() as second_client:
        response = second_client.post(
            "/api/search", json={"query": "vectors stored on disk"}
        )
        assert response.status_code == 200
        result = response.get_json()["data"]
        assert result["count"] >= 1
        assert result["results"][0]["document_name"] == "persistent.txt"

        health = second_client.get("/api/health").get_json()["data"]
        assert health["database"]["documents"] == 1
        assert health["index"]["vectors"] == chunk_count
        assert health["index"]["in_sync"] is True
