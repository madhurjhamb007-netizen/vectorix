"""Helpers for consistent JSON API responses."""

from __future__ import annotations

from typing import Any

from flask import jsonify


def success(data: Any = None, *, status: int = 200, meta: dict[str, Any] | None = None):
    payload: dict[str, Any] = {"data": data}
    if meta is not None:
        payload["meta"] = meta
    return jsonify(payload), status


def error_response(
    message: str,
    *,
    code: str = "bad_request",
    status: int = 400,
    details: dict[str, Any] | None = None,
):
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status
