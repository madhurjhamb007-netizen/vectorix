"""SemanticSearch application entry point.

Run locally:
    python app.py

Or with the Flask CLI (debug reloader):
    flask --app app:app run --debug
"""

from __future__ import annotations

import os

from backend.app_factory import create_app

app = create_app()


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
