"""Tests for overlapping text chunking."""

from __future__ import annotations

import pytest

from backend.models.entities import ExtractedDocument, ExtractedPage
from backend.services.chunking import chunk_document, split_text


def test_short_text_is_one_chunk():
    assert split_text("A short document.", chunk_size=100, overlap=20) == [
        "A short document."
    ]


def test_empty_and_whitespace_text_produce_no_chunks():
    assert split_text("", chunk_size=100, overlap=20) == []
    assert split_text(" \n\t  ", chunk_size=100, overlap=20) == []


def test_chunks_have_overlap_and_stay_within_budget():
    words = [f"word{i:02d}" for i in range(80)]
    text = " ".join(words)

    chunks = split_text(text, chunk_size=100, overlap=24)

    assert len(chunks) > 2
    assert all(len(chunk) <= 100 for chunk in chunks)
    for previous, following in zip(chunks, chunks[1:]):
        previous_words = previous.split()
        following_words = following.split()
        # The next chunk must begin with a suffix of the previous chunk.
        assert following_words[0] in previous_words


def test_leading_overlap_is_actually_present():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

    chunks = split_text(text, chunk_size=35, overlap=12)

    assert len(chunks) >= 2
    first_words = chunks[0].split()
    second_words = chunks[1].split()

    # Find the actual suffix/prefix overlap rather than assuming its length.
    overlap_length = 0
    for length in range(1, min(len(first_words), len(second_words)) + 1):
        if second_words[:length] == first_words[-length:]:
            overlap_length = length
    assert overlap_length >= 1
    assert len(" ".join(second_words[:overlap_length])) <= 12


def test_zero_overlap_does_not_repeat_words():
    text = " ".join(f"token{i}" for i in range(30))

    chunks = split_text(text, chunk_size=45, overlap=0)

    all_words = [word for chunk in chunks for word in chunk.split()]
    assert all_words == text.split()


def test_invalid_chunk_settings_raise():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=-1)


def test_single_word_longer_than_budget_does_not_loop():
    chunks = split_text("supercalifragilisticexpialidocious", chunk_size=10, overlap=2)
    assert chunks == ["supercalifragilisticexpialidocious"]


def test_chunk_document_preserves_pages_and_order():
    document = ExtractedDocument(
        pages=[
            ExtractedPage(page_number=1, text=" ".join(["alpha"] * 80)),
            ExtractedPage(page_number=2, text=" ".join(["beta"] * 80)),
        ],
        page_count=2,
        file_type="pdf",
    )

    chunks = chunk_document(document, chunk_size=100, overlap=20)

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.content for chunk in chunks)
    first_page = [chunk for chunk in chunks if chunk.page_number == 1]
    second_page = [chunk for chunk in chunks if chunk.page_number == 2]
    assert first_page and second_page
    assert all("alpha" in chunk.content for chunk in first_page)
    assert all("beta" in chunk.content for chunk in second_page)


def test_chunk_document_txt_has_no_page_number():
    document = ExtractedDocument(
        pages=[ExtractedPage(page_number=None, text="plain text content " * 20)],
        page_count=0,
        file_type="txt",
    )

    chunks = chunk_document(document, chunk_size=100, overlap=20)

    assert chunks
    assert all(chunk.page_number is None for chunk in chunks)
