"""Report whether the local environment can run SemanticSearch.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --load-model

``--load-model`` also downloads/loads all-MiniLM-L6-v2 and runs a small
encoding. Without the flag, the script only checks imports and storage.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig  # noqa: E402

_DISTRIBUTIONS = {
    "flask": "Flask",
    "numpy": "numpy",
    "pymupdf": "PyMuPDF",
    "faiss": "faiss-cpu",
    "sentence_transformers": "sentence-transformers",
}


def check_import(module: str, label: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        try:
            version = importlib.metadata.version(_DISTRIBUTIONS.get(module, label))
        except importlib.metadata.PackageNotFoundError:
            version = ""
        return True, f"{label} {version}".strip()
    except Exception as exc:  # native DLL failures can raise OSError
        return False, f"{label} unavailable: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SemanticSearch dependencies")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="load the configured embedding model and encode a test sentence",
    )
    args = parser.parse_args()

    print("SemanticSearch environment check")
    print("=" * 42)
    print(f"Python: {sys.version.split()[0]} ({sys.platform})")
    print(f"Project: {PROJECT_ROOT}")

    required_ok = True
    checks = [
        ("flask", "Flask"),
        ("numpy", "NumPy"),
        ("pymupdf", "PyMuPDF"),
    ]
    optional_checks = [
        ("faiss", "FAISS"),
        ("sentence_transformers", "sentence-transformers"),
    ]

    print("\nCore dependencies")
    for module, label in checks:
        ok, message = check_import(module, label)
        print(f"  [{'OK' if ok else 'FAIL'}] {message}")
        required_ok = required_ok and ok

    print("\nOptional native dependencies")
    faiss_ok = True
    st_ok = True
    for module, label in optional_checks:
        ok, message = check_import(module, label)
        if module == "faiss":
            faiss_ok = ok
        elif module == "sentence_transformers":
            st_ok = ok
        status = "OK" if ok else ("FALLBACK" if module == "faiss" else "UNAVAILABLE")
        print(f"  [{status}] {message}")

    try:
        config = AppConfig.from_env()
        config.ensure_directories()
        print("\nStorage")
        print(f"  [OK] {config.storage_dir}")
        print(f"  [OK] database: {config.database_path}")
        print(f"  [OK] index: {config.index_dir}")
        print(f"  [OK] uploads: {config.upload_dir}")
    except Exception as exc:
        print(f"\nStorage configuration failed: {exc}")
        return 1

    print("\nSelected runtime")
    print(f"  Embedding backend: {config.embedding_backend}")
    print(f"  Embedding model:   {config.embedding_model}")
    print(f"  Vector backend:    {'faiss' if faiss_ok and config.vector_backend != 'numpy' else 'numpy'}")
    if not faiss_ok:
        print(
            "  Note: FAISS cannot be imported. The exact NumPy fallback will be "
            "used. Set VECTOR_BACKEND=faiss to make this fatal."
        )
    if not st_ok:
        print(
            "  Note: sentence-transformers/PyTorch cannot be imported. The app "
            "will start, but real semantic search is unavailable. Install a "
            "working PyTorch wheel, or set EMBEDDING_BACKEND=hashing for an "
            "offline, non-semantic demo."
        )

    if args.load_model:
        print("\nModel load test")
        try:
            embedder = __import__(
                "backend.services.embeddings", fromlist=["create_embedder"]
            ).create_embedder(config)
            vector = embedder.encode_query("semantic search runs locally")
            print(f"  [OK] {embedder.name}: vector shape {vector.shape}")
        except Exception as exc:
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            print(
                "  If this is a native DLL/Application Control error, use a "
                "machine that permits PyTorch wheels or set "
                "EMBEDDING_BACKEND=hashing for a non-semantic demo."
            )
            return 1

    print("\nResult:", "core dependencies available" if required_ok else "missing core dependencies")
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
