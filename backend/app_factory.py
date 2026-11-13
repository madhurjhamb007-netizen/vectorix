"""Flask application factory and error handling."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from flask import Flask, g, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from backend import __version__
from backend.container import build_services
from backend.database.connection import initialize_database
from backend.routes import documents_bp, search_bp, system_bp
from backend.services.embeddings import Embedder
from backend.utils.errors import APIError, PayloadTooLargeError
from backend.utils.logging import configure_logging
from backend.utils.responses import error_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

logger = logging.getLogger("semanticsearch.app")


def create_app(config=None, embedder: Embedder | None = None) -> Flask:
    if config is None:
        from config import AppConfig

        config = AppConfig.from_env()

    config.ensure_directories()
    configure_logging(config.log_dir, config.log_level, to_file=config.log_to_file)

    app = Flask(
        __name__.split(".")[0],
        static_folder=str(FRONTEND_DIR / "static"),
        template_folder=str(FRONTEND_DIR / "templates"),
        static_url_path="/static",
    )
    app.config.update(
        SECRET_KEY=config.secret_key,
        # Leave room for multipart overhead; the ingestion service enforces
        # the exact per-file byte limit while streaming.
        MAX_CONTENT_LENGTH=config.max_content_length + 1024 * 1024,
        JSON_SORT_KEYS=False,
        APP_VERSION=__version__,
        SETTINGS=config,
    )

    initialize_database(config.database_path)
    services = build_services(config, embedder=embedder)
    app.extensions["services"] = services

    app.register_blueprint(system_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(search_bp)

    _register_error_handlers(app)
    _register_request_hooks(app, config)

    logger.info(
        "SemanticSearch %s started | storage=%s | index=%s | embedding=%s",
        __version__,
        config.storage_dir,
        services.vector_store.backend_name,
        services.embedder.backend,
    )
    return app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return error_response(
            error.message,
            code=error.code,
            status=error.status_code,
            details=error.details,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(error: RequestEntityTooLarge):
        if request.path.startswith("/api/"):
            limit_mb = app.config["SETTINGS"].max_upload_mb
            return error_response(
                f"File exceeds the {limit_mb} MB upload limit.",
                code=PayloadTooLargeError.code,
                status=413,
            )
        return error

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if request.path.startswith("/api/"):
            return error_response(
                error.description or error.name,
                code=(error.name or "http_error").lower().replace(" ", "_"),
                status=error.code or 500,
            )
        return error

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        logger.exception("Unhandled exception while serving %s", request.path)
        if request.path.startswith("/api/"):
            return error_response(
                "An unexpected server error occurred. Check the application logs.",
                code="internal_error",
                status=500,
            )
        return "Internal Server Error", 500


def _register_request_hooks(app: Flask, config) -> None:
    @app.before_request
    def start_timer():
        g.request_started = time.perf_counter()

    @app.after_request
    def add_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "base-uri 'none'; "
            "form-action 'self'; "
            "frame-ancestors 'none'",
        )
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")

        started = getattr(g, "request_started", None)
        if started is not None and request.path.startswith("/api/"):
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "%s %s -> %s (%.1f ms)",
                request.method,
                request.path,
                response.status_code,
                duration_ms,
            )
        return response


def create_cli_app() -> Flask:
    """Used by ``flask --app app:app run``."""
    return create_app()
