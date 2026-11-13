"""Health, index maintenance, and dashboard page routes."""

from __future__ import annotations

from flask import Blueprint, current_app, render_template

from backend.utils.responses import success

system_bp = Blueprint("system", __name__)


@system_bp.get("/")
def dashboard():
    config = current_app.extensions["services"].config
    return render_template(
        "index.html",
        max_upload_mb=config.max_upload_mb,
        top_k_default=config.top_k_default,
        top_k_max=config.top_k_max,
        allowed_extensions=list(config.allowed_extensions),
        embedding_model=config.embedding_model,
        app_version=current_app.config.get("APP_VERSION", "1.0.0"),
    )


@system_bp.get("/api/health")
def health():
    payload, status_code = current_app.extensions["services"].health.snapshot()
    return success(payload, status=status_code)


@system_bp.post("/api/index/rebuild")
def rebuild_index():
    result = current_app.extensions["services"].documents.rebuild_index()
    return success(result)
