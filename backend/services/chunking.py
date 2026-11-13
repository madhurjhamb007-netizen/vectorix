"""Overlapping, word-aware text chunking."""

from __future__ import annotations

import logging
import re

from backend.models.entities import ChunkPayload, ExtractedDocument

logger = logging.getLogger("semanticsearch.chunking")

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks measured in characters.

    Words are never cut in half.  Each chunk is at most ``chunk_size``
    characters unless a single word is itself longer than the budget.  The
    next chunk starts with up to ``overlap`` characters of trailing context
    from the previous chunk, which preserves sentences that straddle a
    boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = _normalize(text)
    if not text:
        return []

    words = text.split(" ")
    chunks: list[str] = []
    start = 0
    total_words = len(words)

    while start < total_words:
        end = start
        length = 0
        while end < total_words:
            word_length = len(words[end]) + (1 if end > start else 0)
            if end > start and length + word_length > chunk_size:
                break
            length += word_length
            end += 1

        if end == start:  # Single word longer than the whole budget.
            end = start + 1

        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= total_words:
            break

        # Walk backwards to find overlap context, measured in characters.
        new_start = end
        tail_length = 0
        while new_start > start:
            candidate = words[new_start - 1]
            addition = len(candidate) + (1 if tail_length else 0)
            if tail_length + addition > overlap:
                break
            tail_length += addition
            new_start -= 1

        # Guarantee forward progress when overlap <= 0 or the chunk is tiny.
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int,
    overlap: int,
) -> list[ChunkPayload]:
    """Chunk every page independently so page metadata stays accurate."""
    payloads: list[ChunkPayload] = []
    chunk_index = 0

    for page in document.pages:
        for piece in split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            payloads.append(
                ChunkPayload(
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    content=piece,
                )
            )
            chunk_index += 1

    logger.info(
        "Created %s chunks (size=%s, overlap=%s) from a %s document",
        len(payloads),
        chunk_size,
        overlap,
        document.file_type,
    )
    return payloads
