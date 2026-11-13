"""Persistent vector index.

Primary backend: FAISS ``IndexFlatIP`` over L2-normalized vectors, which gives
exact cosine similarity via an inner-product search.  A NumPy implementation
of the same operation is available as a graceful fallback for environments
where the FAISS native extension cannot be imported.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.services.embeddings import normalize_rows
from backend.utils.errors import IndexCompatibilityError, VectorStoreError

logger = logging.getLogger("semanticsearch.vector_store")

MANIFEST_FILE = "manifest.json"
VECTORS_FILE = "vectors.npy"
FAISS_FILE = "semantic.faiss"


class VectorStore:
    def __init__(
        self,
        index_dir: Path | str,
        *,
        backend: str = "auto",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.requested_backend = backend
        self._faiss = self._import_faiss(backend)
        self.backend_name = "faiss" if self._faiss is not None else "numpy"

        self._lock = threading.RLock()
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._chunk_ids: list[str] = []
        self._index = None
        self._embedding_model: str | None = None
        self._load_warning: str | None = None
        self._last_saved_at: str | None = None

        if self.backend_name == "numpy" and backend in {"auto", "faiss"}:
            logger.warning(
                "FAISS is unavailable; using the exact NumPy inner-product fallback."
            )
        self.load()

    # -------------------------------------------------------------- lifecycle
    @staticmethod
    def _import_faiss(backend: str):
        if backend == "numpy":
            return None
        try:
            import faiss  # type: ignore

            return faiss
        except Exception as exc:
            if backend == "faiss":
                raise VectorStoreError(
                    "VECTOR_BACKEND=faiss was requested, but FAISS could not be "
                    f"imported: {exc}"
                ) from exc
            logger.warning("Could not import FAISS (%s). Falling back to NumPy.", exc)
            return None

    @property
    def lock(self) -> threading.RLock:
        """Shared mutation lock used by ingestion, deletion and rebuilding."""
        return self._lock

    @property
    def is_empty(self) -> bool:
        return len(self._chunk_ids) == 0

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.ndim == 2 else 0

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @property
    def chunk_ids(self) -> list[str]:
        with self._lock:
            return list(self._chunk_ids)

    def position_map(self) -> dict[str, int]:
        with self._lock:
            return {chunk_id: position for position, chunk_id in enumerate(self._chunk_ids)}

    def is_compatible(self, embedding_model: str, dimension: int | None = None) -> bool:
        if self.count == 0:
            return True
        if self._embedding_model and embedding_model != self._embedding_model:
            return False
        if dimension is not None and self.dimension and dimension != self.dimension:
            return False
        return True

    def ensure_compatible(
        self, embedding_model: str, dimension: int | None = None
    ) -> None:
        if self.is_compatible(embedding_model, dimension):
            return
        raise IndexCompatibilityError(
            "The saved index was created with a different embedding model or "
            f"dimension ({self._embedding_model}, dim={self.dimension}). "
            "Rebuild the index before adding or searching documents.",
            details={
                "index_model": self._embedding_model,
                "index_dimension": self.dimension,
                "requested_model": embedding_model,
            },
        )

    # ------------------------------------------------------------- persistence
    def load(self) -> None:
        with self._lock:
            manifest_path = self.index_dir / MANIFEST_FILE
            vectors_path = self.index_dir / VECTORS_FILE

            if not manifest_path.exists() and not vectors_path.exists():
                logger.info("No existing vector index found; starting empty.")
                return

            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                chunk_ids = manifest.get("chunk_ids", [])
                model = manifest.get("embedding_model")
                manifest_dimension = int(manifest.get("dimension") or 0)

                if vectors_path.exists():
                    vectors = np.load(vectors_path, allow_pickle=False)
                    vectors = np.asarray(vectors, dtype=np.float32)
                    if vectors.ndim == 1:
                        vectors = vectors.reshape(1, -1)
                else:
                    vectors = np.zeros((0, manifest_dimension), dtype=np.float32)

                if not isinstance(chunk_ids, list) or not all(
                    isinstance(item, str) for item in chunk_ids
                ):
                    raise ValueError("Manifest chunk_ids is not a list of strings")
                if len(set(chunk_ids)) != len(chunk_ids):
                    raise ValueError("Manifest contains duplicate chunk IDs")
                if len(chunk_ids) != vectors.shape[0]:
                    self._load_warning = (
                        "Index manifest and vector snapshot had different row counts; "
                        "the shorter of the two was loaded. Rebuild is recommended."
                    )
                    logger.warning(self._load_warning)
                    count = min(len(chunk_ids), vectors.shape[0])
                    chunk_ids = chunk_ids[:count]
                    vectors = vectors[:count]

                if vectors.size and vectors.shape[1] != (manifest_dimension or vectors.shape[1]):
                    raise ValueError("Vector dimension does not match the manifest")

                self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                self._chunk_ids = list(chunk_ids)
                self._embedding_model = model
                self._last_saved_at = manifest.get("updated_at")

                if self.count:
                    self._index = self._load_or_rebuild_faiss()
                else:
                    self._index = None

                logger.info(
                    "Loaded vector index: %s vectors, dimension=%s, model=%s, backend=%s",
                    self.count,
                    self.dimension,
                    self._embedding_model,
                    self.backend_name,
                )
            except Exception as exc:
                logger.exception("Could not load vector index: %s", exc)
                self._load_warning = (
                    "The persisted vector index could not be read and was reset. "
                    "Rebuild the index from stored chunks."
                )
                self._vectors = np.zeros((0, 0), dtype=np.float32)
                self._chunk_ids = []
                self._index = None
                self._embedding_model = None

    def _load_or_rebuild_faiss(self):
        if self._faiss is None or self.count == 0:
            return None

        faiss_path = self.index_dir / FAISS_FILE
        if faiss_path.exists():
            try:
                index = self._faiss.read_index(str(faiss_path))
                if index.ntotal == self.count and index.d == self.dimension:
                    return index
                logger.warning(
                    "Persisted FAISS index is stale (ntotal=%s, count=%s); "
                    "rebuilding it from the vector snapshot.",
                    index.ntotal,
                    self.count,
                )
            except Exception as exc:
                logger.warning("Could not read FAISS file; rebuilding: %s", exc)

        return self._build_faiss_index(self._vectors)

    def _build_faiss_index(self, vectors: np.ndarray):
        if self._faiss is None or len(vectors) == 0:
            return None
        index = self._faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return index

    @staticmethod
    def _atomic_write_bytes(path: Path, writer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        try:
            writer(Path(temporary))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _save_locked(self) -> None:
        if self.count and self._vectors.ndim != 2:
            raise VectorStoreError("Internal vector store is in an invalid state.")
        if self.count and not self.is_compatible(
            self._embedding_model or "", self.dimension
        ):
            raise VectorStoreError("Refusing to persist an incompatible index.")

        def write_vectors(target: Path) -> None:
            with target.open("wb") as handle:
                np.save(handle, self._vectors, allow_pickle=False)

        try:
            self._atomic_write_bytes(self.index_dir / VECTORS_FILE, write_vectors)

            if self._faiss is not None:
                faiss_path = self.index_dir / FAISS_FILE
                if self.count:
                    index = self._build_faiss_index(self._vectors)
                    self._index = index

                    def write_faiss(target: Path) -> None:
                        self._faiss.write_index(index, str(target))

                    self._atomic_write_bytes(faiss_path, write_faiss)
                elif faiss_path.exists():
                    faiss_path.unlink(missing_ok=True)

            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            manifest = {
                "version": 1,
                "backend": self.backend_name,
                "embedding_model": self._embedding_model,
                "dimension": self.dimension,
                "count": self.count,
                "chunk_ids": self._chunk_ids,
                "updated_at": timestamp,
            }

            def write_manifest(target: Path) -> None:
                target.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            self._atomic_write_bytes(self.index_dir / MANIFEST_FILE, write_manifest)
            self._last_saved_at = timestamp
            self._load_warning = None
        except Exception as exc:
            logger.exception("Failed to persist vector index")
            raise VectorStoreError(f"Could not save the vector index: {exc}") from exc

    # --------------------------------------------------------------- mutations
    def add(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        *,
        embedding_model: str,
    ) -> dict[str, int]:
        """Append vectors atomically and return their index positions."""
        ids = list(chunk_ids)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if len(ids) != matrix.shape[0]:
            raise VectorStoreError("Chunk ID count does not match vector count.")
        if not ids:
            return {}
        if len(set(ids)) != len(ids):
            raise VectorStoreError("Duplicate chunk IDs cannot be indexed.")
        if not np.isfinite(matrix).all():
            raise VectorStoreError("Vectors contain NaN or infinite values.")
        if self.count:
            self.ensure_compatible(embedding_model, int(matrix.shape[1]))
        if len(set(ids) & set(self._chunk_ids)):
            raise VectorStoreError("One or more chunks are already indexed.")

        with self._lock:
            old_vectors = self._vectors
            old_ids = self._chunk_ids
            old_model = self._embedding_model
            old_index = self._index
            old_saved_at = self._last_saved_at
            try:
                normalized = normalize_rows(matrix)
                if self._vectors.size == 0:
                    self._vectors = normalized
                else:
                    self._vectors = np.vstack([self._vectors, normalized])
                self._chunk_ids = old_ids + ids
                self._embedding_model = embedding_model
                # Exact flat index; rebuilding is cheap for local-scale corpora
                # and guarantees the on-disk snapshot and FAISS file agree.
                self._index = self._build_faiss_index(self._vectors)
                self._save_locked()
                return {
                    chunk_id: position
                    for position, chunk_id in enumerate(self._chunk_ids)
                    if chunk_id in set(ids)
                }
            except Exception:
                self._vectors = old_vectors
                self._chunk_ids = old_ids
                self._embedding_model = old_model
                self._index = old_index
                self._last_saved_at = old_saved_at
                raise

    def remove(self, chunk_ids: Sequence[str]) -> int:
        """Remove chunk IDs, rebuild the flat index, and persist."""
        to_remove = set(chunk_ids)
        with self._lock:
            if not to_remove:
                return 0
            keep = [chunk_id not in to_remove for chunk_id in self._chunk_ids]
            removed = len(self._chunk_ids) - sum(keep)
            if removed == 0:
                return 0

            old_vectors = self._vectors
            old_ids = self._chunk_ids
            old_index = self._index
            old_model = self._embedding_model
            old_saved_at = self._last_saved_at
            try:
                self._vectors = self._vectors[np.asarray(keep, dtype=bool)]
                if self._vectors.shape[0] == 0:
                    self._vectors = np.zeros((0, 0), dtype=np.float32)
                    self._embedding_model = None
                self._chunk_ids = [
                    chunk_id for chunk_id, kept in zip(old_ids, keep) if kept
                ]
                self._index = self._build_faiss_index(self._vectors)
                self._save_locked()
                return removed
            except Exception:
                self._vectors = old_vectors
                self._chunk_ids = old_ids
                self._index = old_index
                self._embedding_model = old_model
                self._last_saved_at = old_saved_at
                raise

    def replace_all(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        *,
        embedding_model: str,
    ) -> None:
        """Atomically replace the complete index (used by rebuilds)."""
        ids = list(chunk_ids)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            matrix = matrix.reshape(-1, matrix.shape[-1]) if matrix.size else np.zeros((0, 0), np.float32)
        if len(ids) != matrix.shape[0]:
            raise VectorStoreError("Chunk ID count does not match vector count.")
        if len(set(ids)) != len(ids):
            raise VectorStoreError("Duplicate chunk IDs cannot be indexed.")
        if matrix.size and not np.isfinite(matrix).all():
            raise VectorStoreError("Vectors contain NaN or infinite values.")

        with self._lock:
            old_vectors = self._vectors
            old_ids = self._chunk_ids
            old_index = self._index
            old_model = self._embedding_model
            old_saved_at = self._last_saved_at
            try:
                normalized = normalize_rows(matrix) if len(ids) else np.zeros((0, 0), dtype=np.float32)
                self._vectors = normalized
                self._chunk_ids = ids
                self._embedding_model = embedding_model if ids else None
                self._index = self._build_faiss_index(normalized)
                self._save_locked()
            except Exception:
                self._vectors = old_vectors
                self._chunk_ids = old_ids
                self._index = old_index
                self._embedding_model = old_model
                self._last_saved_at = old_saved_at
                raise

    # ------------------------------------------------------------------ search
    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` pairs, best first."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        with self._lock:
            if not self._chunk_ids:
                return []
            if query.shape[0] != self.dimension:
                raise VectorStoreError(
                    f"Query dimension {query.shape[0]} does not match index "
                    f"dimension {self.dimension}. Rebuild the index with the "
                    "current embedding model."
                )
            if not np.isfinite(query).all():
                raise VectorStoreError("Query vector contains invalid values.")
            if np.linalg.norm(query) == 0:
                return []

            query = normalize_rows(query.reshape(1, -1))[0]
            k = min(int(top_k), self.count)

            if self.backend_name == "faiss" and self._index is not None:
                scores, indices = self._index.search(query.reshape(1, -1), k)
                pairs = [
                    (self._chunk_ids[int(position)], float(score))
                    for score, position in zip(scores[0], indices[0])
                    if int(position) >= 0
                ]
            else:
                similarities = self._vectors @ query
                top_positions = np.argsort(-similarities)[:k]
                pairs = [
                    (self._chunk_ids[int(position)], float(similarities[position]))
                    for position in top_positions
                ]

            for chunk_id, score in pairs:
                if not np.isfinite(score):
                    raise VectorStoreError("Search produced a non-finite score.")
            return pairs

    # -------------------------------------------------------------- diagnostics
    def describe(self) -> dict:
        with self._lock:
            return {
                "backend": self.backend_name,
                "requested_backend": self.requested_backend,
                "faiss_available": self._faiss is not None,
                "count": self.count,
                "dimension": self.dimension,
                "embedding_model": self._embedding_model,
                "last_saved_at": self._last_saved_at,
                "load_warning": self._load_warning,
                "files": {
                    "manifest": (self.index_dir / MANIFEST_FILE).exists(),
                    "vectors": (self.index_dir / VECTORS_FILE).exists(),
                    "faiss": (self.index_dir / FAISS_FILE).exists(),
                },
            }
