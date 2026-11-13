"""Composition root: wires config, persistence, and services together."""

from __future__ import annotations

from dataclasses import dataclass

from backend import __version__
from backend.database.repositories import DocumentRepository
from backend.services.documents import DocumentService
from backend.services.embeddings import Embedder, create_embedder
from backend.services.health import HealthService
from backend.services.ingestion import IngestionService
from backend.services.search import SearchService
from backend.services.vector_store import VectorStore


@dataclass
class ServiceContainer:
    config: object
    repository: DocumentRepository
    embedder: Embedder
    vector_store: VectorStore
    ingestion: IngestionService
    documents: DocumentService
    search: SearchService
    health: HealthService


def build_services(config, embedder: Embedder | None = None) -> ServiceContainer:
    repository = DocumentRepository(config.database_path)
    resolved_embedder = embedder or create_embedder(config)
    vector_store = VectorStore(config.index_dir, backend=config.vector_backend)

    ingestion = IngestionService(
        config=config,
        repository=repository,
        embedder=resolved_embedder,
        vector_store=vector_store,
    )
    documents = DocumentService(
        config=config,
        repository=repository,
        embedder=resolved_embedder,
        vector_store=vector_store,
    )
    search = SearchService(
        config=config,
        repository=repository,
        embedder=resolved_embedder,
        vector_store=vector_store,
    )
    health = HealthService(
        repository=repository,
        embedder=resolved_embedder,
        vector_store=vector_store,
        version=__version__,
    )

    return ServiceContainer(
        config=config,
        repository=repository,
        embedder=resolved_embedder,
        vector_store=vector_store,
        ingestion=ingestion,
        documents=documents,
        search=search,
        health=health,
    )
