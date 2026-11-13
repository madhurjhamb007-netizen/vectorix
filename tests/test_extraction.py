"""Tests for TXT/PDF extraction."""

from __future__ import annotations

import pytest

from backend.services.extraction import extract_document, extract_pdf, extract_txt
from backend.utils.errors import EmptyDocumentError, ExtractionError, ValidationError
from tests.support import make_pdf_bytes


def test_extract_txt_normalizes_line_endings(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"First line\r\nSecond\tline\r\n")

    result = extract_txt(path)

    assert result.file_type == "txt"
    assert result.page_count == 0
    assert result.pages[0].page_number is None
    assert "First line\nSecond" in result.pages[0].text


def test_extract_txt_supports_utf8_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes("\ufeffCafé résumé".encode("utf-8"))

    result = extract_txt(path)

    assert "Café résumé" in result.pages[0].text


def test_extract_txt_supports_utf16_bom(tmp_path):
    path = tmp_path / "utf16.txt"
    path.write_bytes("Résumé with unicode".encode("utf-16"))

    result = extract_txt(path)

    assert result.pages[0].text == "Résumé with unicode"


def test_extract_txt_supports_utf32_bom(tmp_path):
    path = tmp_path / "utf32.txt"
    path.write_bytes("Unicode text ☃".encode("utf-32"))

    result = extract_txt(path)

    assert result.pages[0].text == "Unicode text ☃"


def test_extract_txt_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes("Price €5".encode("cp1252"))

    result = extract_txt(path)

    assert "Price €5" in result.pages[0].text


def test_extract_txt_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"   \n\t\n")

    with pytest.raises(EmptyDocumentError):
        extract_txt(path)


def test_extract_txt_rejects_binary_content(tmp_path):
    path = tmp_path / "binary.txt"
    path.write_bytes(b"hello\x00world")

    with pytest.raises(ExtractionError):
        extract_txt(path)


def test_extract_txt_rejects_pdf_masquerading_as_txt(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(make_pdf_bytes(["not really txt"]))

    with pytest.raises(ValidationError):
        extract_txt(path)


def test_extract_pdf_preserves_page_numbers(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf_bytes(["Alpha page content", "Beta page content"]))

    result = extract_pdf(path)

    assert result.file_type == "pdf"
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Alpha" in result.pages[0].text
    assert "Beta" in result.pages[1].text


def test_extract_pdf_skips_blank_pages_but_keeps_real_page_numbers(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf_bytes(["", "Only this page has text"]))

    result = extract_pdf(path)

    assert result.page_count == 2
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 2


def test_extract_pdf_with_no_text_raises(tmp_path):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(make_pdf_bytes(["", ""]))

    with pytest.raises(EmptyDocumentError):
        extract_pdf(path)


def test_corrupt_pdf_raises_extraction_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is definitely not a valid pdf")

    with pytest.raises(ExtractionError):
        extract_pdf(path)


def test_non_pdf_with_pdf_extension_raises(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"plain text pretending to be a pdf")

    with pytest.raises(ExtractionError):
        extract_pdf(path)


def test_extract_document_dispatch(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("Hello world", encoding="utf-8")

    assert extract_document(path, "txt").pages[0].text == "Hello world"

    with pytest.raises(ValidationError):
        extract_document(path, "docx")
