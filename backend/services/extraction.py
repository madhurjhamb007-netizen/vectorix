"""Text extraction for PDF and TXT documents."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.models.entities import ExtractedDocument, ExtractedPage
from backend.utils.errors import EmptyDocumentError, ExtractionError, ValidationError

logger = logging.getLogger("semanticsearch.extraction")

try:  # PyMuPDF >= 1.24 exposes the modern module name.
    import pymupdf as _pymupdf
except ImportError:  # pragma: no cover - fallback for older installs
    try:
        import fitz as _pymupdf  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - dependency installation issue
        _pymupdf = None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip NUL and other control characters that break tokenisation/JSON.
    return "".join(
        char for char in text if char in "\n\t" or (char.isprintable() and char != "\x00")
    ).strip()


def _decode_txt(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        raise ValidationError("This file looks like a PDF but was uploaded as .txt.")

    encodings = ["utf-8-sig"]
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings.insert(0, "utf-32")
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(0, "utf-16")

    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        try:
            text = data.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ExtractionError(
                "The TXT file is not valid UTF-8, UTF-16 or Windows-1252 text."
            ) from exc

    if "\x00" in text:
        raise ExtractionError("The file does not appear to be a plain-text document.")
    return _normalize_text(text)


def extract_txt(path: Path) -> ExtractedDocument:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ExtractionError(f"Could not read the TXT file: {exc}") from exc

    if not data.strip():
        raise EmptyDocumentError("The TXT file is empty.")

    text = _decode_txt(data)
    if not text:
        raise EmptyDocumentError("No readable text could be extracted from the TXT file.")

    logger.info("Extracted %s characters from TXT file %s", len(text), Path(path).name)
    return ExtractedDocument(
        pages=[ExtractedPage(page_number=None, text=text)],
        page_count=0,
        file_type="txt",
    )


def extract_pdf(path: Path) -> ExtractedDocument:
    if _pymupdf is None:  # pragma: no cover
        raise ExtractionError(
            "PDF support is unavailable because PyMuPDF is not installed. "
            "Run: pip install pymupdf"
        )

    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"Could not read the PDF file: {exc}") from exc

    if not data.startswith(b"%PDF-"):
        raise ExtractionError("The uploaded file is not a valid PDF (missing PDF header).")

    document = None
    try:
        # Load from memory so the temporary upload file is never kept open on
        # Windows. PyMuPDF can hold a file handle after a parsing failure.
        document = _pymupdf.open(stream=data, filetype="pdf")
        if document.needs_pass:
            raise ExtractionError("Password-protected PDFs are not supported.")
        if not document.is_pdf:
            raise ExtractionError("The uploaded file is not a valid PDF.")

        pages: list[ExtractedPage] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = _normalize_text(page.get_text("text"))
            if text:
                pages.append(ExtractedPage(page_number=page_index + 1, text=text))

        page_count = int(document.page_count)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(
            "The PDF could not be parsed. It may be corrupted or malformed."
        ) from exc
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:  # pragma: no cover
                pass

    if not pages:
        raise EmptyDocumentError(
            "No extractable text was found in the PDF. Scanned/image-only PDFs "
            "require OCR, which is not enabled in this project."
        )

    logger.info(
        "Extracted %s pages (%s chars) from PDF %s",
        page_count,
        sum(len(page.text) for page in pages),
        path.name,
    )
    return ExtractedDocument(pages=pages, page_count=page_count, file_type="pdf")


def extract_document(path: Path, extension: str) -> ExtractedDocument:
    """Dispatch extraction based on a validated lowercase extension."""
    extension = extension.lower().lstrip(".")
    if extension == "txt":
        return extract_txt(path)
    if extension == "pdf":
        return extract_pdf(path)
    raise ValidationError(f"Unsupported file type: .{extension}")
