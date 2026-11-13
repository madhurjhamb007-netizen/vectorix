"""Application-level exceptions mapped to HTTP responses by the API layer."""

from __future__ import annotations

from typing import Any


class APIError(Exception):
    """Base class for expected, user-facing errors."""

    status_code = 400
    code = "bad_request"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(APIError):
    status_code = 400
    code = "validation_error"


class NotFoundError(APIError):
    status_code = 404
    code = "not_found"


class ConflictError(APIError):
    status_code = 409
    code = "conflict"


class PayloadTooLargeError(APIError):
    status_code = 413
    code = "file_too_large"


class ExtractionError(APIError):
    status_code = 422
    code = "extraction_failed"


class EmptyDocumentError(ExtractionError):
    code = "empty_document"


class EmbeddingError(APIError):
    status_code = 503
    code = "embedding_unavailable"


class IndexCompatibilityError(APIError):
    status_code = 409
    code = "index_incompatible"


class VectorStoreError(APIError):
    status_code = 500
    code = "index_error"
