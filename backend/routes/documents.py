"""Document management API endpoints."""

from __future__ import annotations

from flask import Blueprint, Request, current_app, request

from backend.utils.errors import ValidationError
from backend.utils.responses import success

documents_bp = Blueprint("documents_api", __name__, url_prefix="/api/documents")


def _services():
    return current_app.extensions["services"]


@documents_bp.get("")
def list_documents():
    documents = _services().documents.list_documents()
    return success([document.to_dict() for document in documents], meta={"count": len(documents)})


@documents_bp.post("/upload")
def upload_document():
    upload = request.files.get("file")
    if upload is None:
        # Support the plural field for simple API clients while keeping one
        # document per request (and therefore one atomic transaction).
        uploads = request.files.getlist("files")
        upload = uploads[0] if uploads else None
    if upload is None:
        raise ValidationError("No file was provided. Use the 'file' form field.")

    document = _services().ingestion.ingest(upload)
    response, status = success(document.to_dict(), status=201)
    response.headers["Location"] = f"/api/documents/{document.id}"
    return response, status


@documents_bp.get("/<document_id>")
def get_document(document_id: str):
    document = _services().documents.get_document(document_id)
    return success(document.to_dict())


@documents_bp.delete("/<document_id>")
def delete_document(document_id: str):
    result = _services().documents.delete_document(document_id)
    return success(result)
