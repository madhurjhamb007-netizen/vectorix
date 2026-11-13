"""Document deletion must remove vectors from future search results."""

from __future__ import annotations

from tests.support import upload_pdf, upload_txt


def data(response):
    return response.get_json()["data"]


def test_deleting_document_removes_chunks_vectors_and_file(client, services):
    supervised = data(
        upload_txt(
            client,
            "Supervised learning uses labeled examples for classification and regression.",
            "supervised.txt",
        )
    )
    remaining = data(
        upload_txt(
            client,
            "FAISS retrieves nearest vectors using cosine similarity.",
            "vectors.txt",
        )
    )

    initial = client.post("/api/search", json={"query": "classification labels"}).get_json()["data"]
    assert initial["results"][0]["document_id"] == supervised["id"]
    initial_vector_count = services.vector_store.count
    stored = services.repository.get_document(supervised["id"])
    stored_path = services.config.upload_dir / stored.stored_name

    response = client.delete(f"/api/documents/{supervised['id']}")
    assert response.status_code == 200

    result = data(response)
    assert result["chunks_removed"] == supervised["chunk_count"]
    assert result["vectors_removed"] == supervised["chunk_count"]
    assert services.vector_store.count == initial_vector_count - supervised["chunk_count"]
    assert not stored_path.exists()

    after = client.post("/api/search", json={"query": "classification labels"}).get_json()["data"]
    assert all(item["document_id"] != supervised["id"] for item in after["results"])
    assert all(item["document_name"] != "supervised.txt" for item in after["results"])

    # The remaining document still works.
    vector_search = client.post(
        "/api/search", json={"query": "nearest vectors similarity"}
    ).get_json()["data"]
    assert any(item["document_id"] == remaining["id"] for item in vector_search["results"])

    health = client.get("/api/health").get_json()["data"]
    assert health["index"]["in_sync"] is True
    assert health["index"]["vectors"] == health["database"]["chunks"]


def test_remaining_chunk_vector_positions_match_the_index(client, services):
    first = data(upload_txt(client, "Supervised learning with labels.", "a.txt"))
    data(upload_txt(client, "Unsupervised clustering with unlabeled data.", "b.txt"))

    client.delete(f"/api/documents/{first['id']}")

    chunks = services.repository.get_all_chunks()
    positions = services.vector_store.position_map()
    assert chunks
    assert len(chunks) == len(positions)
    for chunk in chunks:
        assert chunk.vector_position == positions[chunk.id]


def test_deleting_pdf_document_removes_all_page_chunks(client, services):
    document = data(
        upload_pdf(
            client,
            ["Page one about supervised labels.", "Page two about clustering groups."],
            "pages.pdf",
        )
    )
    assert document["chunk_count"] >= 2

    client.delete(f"/api/documents/{document['id']}")

    assert services.repository.count_chunks() == 0
    assert services.vector_store.count == 0
    result = client.post("/api/search", json={"query": "clustering"}).get_json()["data"]
    assert result["reason"] == "no_documents"
    assert result["results"] == []


def test_deleting_one_of_many_keeps_index_in_sync(client):
    for index in range(3):
        data(
            upload_txt(
                client,
                f"Document {index} discusses unsupervised clustering and unlabeled groups.",
                f"doc-{index}.txt",
            )
        )

    docs = client.get("/api/documents").get_json()["data"]
    victim = docs[1]
    client.delete(f"/api/documents/{victim['id']}")

    health = client.get("/api/health").get_json()["data"]
    assert health["database"]["documents"] == 2
    assert health["index"]["vectors"] == health["database"]["chunks"]
    assert health["index"]["in_sync"] is True
