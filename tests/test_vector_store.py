"""Tests for the persistent vector index and cosine search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.vector_store import FAISS_FILE, MANIFEST_FILE, VECTORS_FILE, VectorStore
from backend.utils.errors import IndexCompatibilityError, VectorStoreError


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "indexes", backend="numpy")


def test_add_and_search_returns_cosine_matches(store):
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    positions = store.add(["a", "b", "c"], vectors, embedding_model="test-model")

    results = store.search(np.array([1.0, 0.0, 0.0]), top_k=2)

    assert positions == {"a": 0, "b": 1, "c": 2}
    assert [chunk_id for chunk_id, _ in results] == ["a", "c"]
    assert results[0][1] > results[1][1] > 0
    assert store.count == 3
    assert store.dimension == 3
    assert store.embedding_model == "test-model"


def test_search_caps_top_k_at_index_size(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")
    assert len(store.search(np.array([1.0, 0.0]), top_k=99)) == 1


def test_index_persists_and_reloads(store, tmp_path):
    store.add(
        ["a", "b"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedding_model="test-model",
    )

    reloaded = VectorStore(tmp_path / "indexes", backend="numpy")
    assert reloaded.count == 2
    assert reloaded.dimension == 2
    assert reloaded.embedding_model == "test-model"
    assert reloaded.chunk_ids == ["a", "b"]

    results = reloaded.search(np.array([0.0, 1.0]), top_k=1)
    assert results[0][0] == "b"
    assert results[0][1] == pytest.approx(1.0)


def test_manifest_maps_positions_to_chunk_ids(store):
    store.add(["x", "y"], np.array([[1.0, 0.0], [0.0, 1.0]]), embedding_model="m")
    manifest = json.loads((store.index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["chunk_ids"] == ["x", "y"]
    assert manifest["count"] == 2
    assert manifest["embedding_model"] == "m"
    assert (store.index_dir / VECTORS_FILE).exists()


def test_remove_deletes_vectors_and_compacts_positions(store):
    store.add(
        ["a", "b", "c"],
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        embedding_model="m",
    )

    removed = store.remove(["b"])

    assert removed == 1
    assert store.chunk_ids == ["a", "c"]
    assert store.position_map() == {"a": 0, "c": 1}
    result_ids = [chunk_id for chunk_id, _ in store.search(np.array([0.0, 1.0]), 5)]
    assert "b" not in result_ids
    assert store.count == 2


def test_removing_unknown_id_is_a_noop(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")
    assert store.remove(["does-not-exist"]) == 0
    assert store.count == 1


def test_remove_everything_clears_model(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")
    store.remove(["a"])
    assert store.is_empty
    assert store.embedding_model is None
    assert store.search(np.array([1.0, 0.0]), 3) == []


def test_replace_all_swaps_index(store):
    store.add(["old"], np.array([[1.0, 0.0]]), embedding_model="m")
    store.replace_all(
        ["new-1", "new-2"],
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        embedding_model="m2",
    )
    assert store.chunk_ids == ["new-1", "new-2"]
    assert store.embedding_model == "m2"
    assert store.search(np.array([0.0, 1.0]), 1)[0][0] == "new-1"


def test_empty_index_search_is_safe(store):
    assert store.search(np.array([1.0, 2.0]), top_k=5) == []
    assert store.count == 0


def test_query_dimension_mismatch_raises(store):
    store.add(["a"], np.array([[1.0, 0.0, 0.0]]), embedding_model="m")
    with pytest.raises(VectorStoreError, match="dimension"):
        store.search(np.array([1.0, 0.0]), top_k=1)


def test_duplicate_chunk_ids_rejected(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")
    with pytest.raises(VectorStoreError, match="already indexed"):
        store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")


def test_mismatched_vector_count_rejected(store):
    with pytest.raises(VectorStoreError, match="does not match"):
        store.add(["a", "b"], np.array([[1.0, 0.0]]), embedding_model="m")


def test_compatibility_guard_for_different_models(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="model-a")
    with pytest.raises(IndexCompatibilityError):
        store.ensure_compatible("model-b")
    store.ensure_compatible("model-a")
    empty_store = VectorStore(store.index_dir.parent / "empty", backend="numpy")
    empty_store.ensure_compatible("anything")


def test_corrupt_manifest_resets_index_with_warning(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / MANIFEST_FILE).write_text("{not valid json", encoding="utf-8")
    store = VectorStore(directory, backend="numpy")
    assert store.count == 0
    assert store.describe()["load_warning"] is not None


def test_describe_reports_backend_and_files(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")
    info = store.describe()
    assert info["backend"] == "numpy"
    assert info["faiss_available"] is False
    assert info["files"]["manifest"] is True
    assert info["files"]["vectors"] is True


def test_saved_faiss_file_is_not_expected_in_numpy_mode(store):
    store.add(["a"], np.array([[1.0, 0.0]]), embedding_model="m")
    assert not (store.index_dir / FAISS_FILE).exists()
