"""File-system helpers with path-traversal protections."""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path

from backend.utils.errors import ValidationError

_SAFE_EXTENSION = re.compile(r"^[a-z0-9]{1,10}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_original_filename(filename: str | None) -> str:
    """Return a display-only filename stripped of any directory components.

    The returned value is never used to build a filesystem path.  It is stored
    in SQLite and rendered as text in the UI.
    """
    if not filename or not filename.strip():
        raise ValidationError("A filename is required.")

    # Handle both Windows and POSIX separators, including crafted names.
    basename = filename.replace("\\", "/").split("/")[-1].strip()
    basename = unicodedata.normalize("NFC", basename)
    basename = _CONTROL_CHARS.sub("", basename)

    if basename in {"", ".", ".."}:
        raise ValidationError("Invalid filename.")
    if len(basename) > 255:
        stem, dot, ext = basename.rpartition(".")
        basename = (stem[: 255 - len(ext) - 1] + dot + ext) if dot else basename[:255]

    return basename


def get_extension(filename: str) -> str:
    """Return the lowercase extension without a leading dot."""
    suffix = Path(filename).suffix.lower().lstrip(".")
    if not suffix or not _SAFE_EXTENSION.match(suffix):
        raise ValidationError("The file must have a valid .pdf or .txt extension.")
    return suffix


def make_stored_filename(extension: str) -> str:
    """Generate a collision-resistant, user-independent filename."""
    ext = extension.lower().lstrip(".")
    if not _SAFE_EXTENSION.match(ext):
        raise ValidationError("Invalid file extension.")
    return f"{uuid.uuid4().hex}.{ext}"


def ensure_within_directory(path: Path, directory: Path) -> Path:
    """Resolve *path* and assert it remains inside *directory*."""
    resolved = path.resolve()
    root = directory.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidationError("Invalid file path.") from exc
    return resolved


def safe_unlink(path: Path, logger=None) -> bool:
    """Delete a file if it exists, never raising to the caller."""
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as exc:  # pragma: no cover - platform dependent
        if logger:
            logger.warning("Could not delete file %s: %s", path, exc)
        return False


def human_file_size(size_bytes: int) -> str:
    """Format a byte count for logs / CLI output."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
