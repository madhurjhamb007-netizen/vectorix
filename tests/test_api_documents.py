"""API tests for upload, listing, validation, and security properties."""

from __future__ import annotations

from pathlib import Path

from backend.app_factory import create_app
from backend.utils.files import sanitize_original_filename
from config import AppConfig
from tests.support import (
    make_pdf_bytes,
    upload_bytes,
    upload_pdf,
    upload_txt,
)


def body(response):
    payload = response.get_json()
    assert payload is not None
    return payload


def test_health_endpoint_reports_empty_state(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = body(response)["data"]
    assert data["status"] == "ok"
    assert data["database"] == {"connected": True, "documents": 0, "chunks": 0}
    assert data["index"]["in_sync"] is True
    assert data["index"]["vectors"] == 0
    assert data["embedding"]["model"] == "test-topic-embedder"


def test_dashboard_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SemanticSearch" in response.data
    assert b"Add documents" in response.data
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_upload_txt_returns_created_document(client, services):
    response = upload_txt(
        client,
        "Supervised learning uses labeled data for classification.",
        filename="ml-notes.txt",
    )

    assert response.status_code == 201
    document = body(response)["data"]
    assert document["name"] == "ml-notes.txt"
    assert document["chunk_count"] >= 1
    assert document["page_count"] == 0
    assert document["status"] == "ready"
    assert response.headers["Location"] == f"/api/documents/{document['id']}"

    stored = services.repository.get_document(document["id"])
    assert stored is not None
    stored_path = services.config.upload_dir / stored.stored_name
    assert stored_path.exists()
    assert stored.stored_name.endswith(".txt")
    assert stored.stored_name != "ml-notes.txt"


def test_upload_pdf_preserves_page_count(client, services):
    response = upload_pdf(
        client,
        ["Supervised learning on page one.", "Unsupervised clustering on page two."],
        filename="paper.pdf",
    )

    assert response.status_code == 201
    document = body(response)["data"]
    assert document["mime_type"] == "application/pdf"
    assert document["page_count"] == 2
    assert document["chunk_count"] >= 2
    assert services.vector_store.count == document["chunk_count"]


def test_list_and_get_documents(client):
    first = body(upload_txt(client, "First document", "first.txt"))["data"]
    body(upload_txt(client, "Second document", "second.txt"))

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    payload = body(listing)
    assert payload["meta"]["count"] == 2
    assert len(payload["data"]) == 2

    detail = client.get(f"/api/documents/{first['id']}")
    assert detail.status_code == 200
    assert body(detail)["data"]["name"] == "first.txt"

    missing = client.get("/api/documents/" + "0" * 32)
    assert missing.status_code == 404


def test_duplicate_upload_is_rejected(client):
    content = "A document about vector similarity search."
    first = upload_txt(client, content, "notes.txt")
    assert first.status_code == 201

    duplicate = upload_txt(client, content, "same-content-different-name.txt")

    assert duplicate.status_code == 409
    error = body(duplicate)["error"]
    assert error["code"] == "conflict"
    assert error["details"]["document_id"] == body(first)["data"]["id"]
    assert len(client.get("/api/documents").get_json()["data"]) == 1


def test_unsupported_extension_is_rejected(client):
    response = upload_bytes(client, b"some data", "notes.docx")

    assert response.status_code == 400
    assert body(response)["error"]["code"] == "validation_error"


def test_missing_file_is_rejected(client):
    response = client.post("/api/documents/upload", data={})
    assert response.status_code == 400
    assert "file" in body(response)["error"]["message"].lower()


def test_empty_file_is_rejected(client):
    response = upload_txt(client, b"", "empty.txt")
    assert response.status_code == 400
    assert body(response)["error"]["code"] == "validation_error"


def test_binary_txt_is_rejected(client):
    response = upload_bytes(client, b"hello\x00world", "binary.txt")
    assert response.status_code == 422
    assert body(response)["error"]["code"] == "extraction_failed"


def test_corrupted_pdf_is_rejected_and_leaves_no_files(client, services):
    response = upload_bytes(
        client, b"%PDF-1.7\nthis is not a real PDF", "broken.pdf"
    )

    assert response.status_code == 422
    assert body(response)["error"]["code"] == "extraction_failed"
    assert client.get("/api/documents").get_json()["data"] == []
    assert services.vector_store.count == 0
    leftover = [
        path
        for path in services.config.upload_dir.iterdir()
        if path.is_file()
    ]
    assert leftover == []


def test_text_pdf_with_no_extractable_text_is_rejected(client):
    response = upload_pdf(client, ["", ""], "scanned.pdf")
    assert response.status_code == 422
    assert body(response)["error"]["code"] == "empty_document"


def test_oversized_file_is_rejected(tmp_path, embedder):
    config = AppConfig.for_testing(tmp_path / "large", max_content_length=4096)
    app = create_app(config, embedder=embedder)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        response = upload_txt(test_client, b"x" * 5000, "large.txt")

    assert response.status_code == 413
    assert body(response)["error"]["code"] == "file_too_large"


def test_path_traversal_filename_is_sanitized(client, services):
    response = upload_txt(
        client,
        "Path traversal attempt.",
        filename="../../../../evil.txt",
    )

    assert response.status_code == 201
    document = body(response)["data"]
    assert document["name"] == "evil.txt"

    stored = services.repository.get_document(document["id"])
    stored_path = services.config.upload_dir / stored.stored_name
    assert stored_path.resolve().parent == services.config.upload_dir.resolve()
    assert not (services.config.storage_dir.parent / "evil.txt").exists()
    assert not (Path(services.config.storage_dir).parent / "evil.txt").exists()


def test_windows_style_path_traversal_filename_is_sanitized(client):
    response = upload_txt(client, "windows path", filename="..\\..\\notes.txt")
    assert response.status_code == 201
    assert body(response)["data"]["name"] == "notes.txt"
    assert sanitize_original_filename("..\\..\\notes.txt") == "notes.txt"


def test_invalid_document_id_and_delete_errors(client):
    invalid = client.delete("/api/documents/not-a-uuid")
    assert invalid.status_code == 400

    missing = client.delete("/api/documents/" + "a" * 32)
    assert missing.status_code == 404


def test_delete_removes_stored_file_and_metadata(client, services):
    document = body(upload_txt(client, "Delete me from the index.", "delete-me.txt"))["data"]
    stored = services.repository.get_document(document["id"])
    stored_path = services.config.upload_dir / stored.stored_name
    assert stored_path.exists()

    response = client.delete(f"/api/documents/{document['id']}")

    assert response.status_code == 200
    result = body(response)["data"]
    assert result["deleted"] is True
    assert result["chunks_removed"] >= 1
    assert result["vectors_removed"] >= 1
    assert result["warnings"] == []
    assert not stored_path.exists()
    assert services.repository.get_document(document["id"]) is None
    assert services.vector_store.count == 0

    again = client.delete(f"/api/documents/{document['id']}")
    assert again.status_code == 404


def test_api_404_returns_json(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.is_json
    assert "error" in response.get_json()


def test_pdf_upload_bytes_are_valid_helper():
    data = make_pdf_bytes(["hello", "world"])
    assert data.startswith(b"%PDF-")
