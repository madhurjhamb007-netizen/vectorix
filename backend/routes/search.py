"""Semantic search API endpoint."""

from __future__ import annotations

from flask import Blueprint, current_app, request

from backend.utils.errors import ValidationError
from backend.utils.responses import success

search_bp = Blueprint("search_api", __name__, url_prefix="/api/search")


@search_bp.post("")
def semantic_search():
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValidationError("Request body must be JSON with a non-empty 'query'.")
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    result = current_app.extensions["services"].search.search(
        payload.get("query"), payload.get("top_k")
    )
    return success(result)
