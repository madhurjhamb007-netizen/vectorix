"""Pytest fixtures for isolated application instances."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app_factory import create_app  # noqa: E402
from config import AppConfig  # noqa: E402
from tests.support import TopicEmbedder  # noqa: E402


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    return AppConfig.for_testing(tmp_path)


@pytest.fixture
def embedder() -> TopicEmbedder:
    return TopicEmbedder()


@pytest.fixture
def app(test_config: AppConfig, embedder: TopicEmbedder):
    application = create_app(test_config, embedder=embedder)
    application.config.update(TESTING=True)
    yield application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def services(app):
    return app.extensions["services"]
