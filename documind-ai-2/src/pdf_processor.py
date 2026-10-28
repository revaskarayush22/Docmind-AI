"""PDF text extraction (PyMuPDF), page-aware, with clear failure modes.

PDFs are parsed entirely in memory from the uploaded bytes; no file is written to
disk, so user-controlled filenames never touch the filesystem.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3 exposes the "pymupdf" name
except ImportError:  # pragma: no cover - older releases
    import fitz  # type: ignore


class PDFProcessingError(Exception):
    """Base class. ``str(exc)`` is safe to show to end users."""


class CorruptedPDFError(PDFProcessingError):
    pass


class EncryptedPDFError(PDFProcessingError):
    pass


class NoExtractableTextError(PDFProcessingError):
    pass


class TooManyPagesError(PDFProcessingError):
    pass


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based, matches what a PDF viewer shows
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    pages: list[PageText]
    total_pages: int
    empty_pages: int  # pages with no extractable text (possibly scanned images)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize extracted text while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated at line ends
    # Single newlines inside a paragraph become spaces; blank lines stay as breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def extract_pages(data: bytes, max_pages: int = 200, min_chars: int = 20) -> ExtractedDocument:
    """Extract cleaned text per page.

    Raises:
        CorruptedPDFError, EncryptedPDFError, TooManyPagesError, NoExtractableTextError
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for bad input
        raise CorruptedPDFError(
            "This file could not be opened as a PDF. It may be corrupted or not a real PDF."
        ) from exc

    try:
        if doc.needs_pass or doc.is_encrypted:
            raise EncryptedPDFError(
                "This PDF is password-protected. Remove the password and upload it again."
            )
        total = doc.page_count
        if total == 0:
            raise CorruptedPDFError("This PDF has no pages.")
        if total > max_pages:
            raise TooManyPagesError(
                f"This PDF has {total} pages; the limit is {max_pages}. Split it and upload the parts."
            )
        pages: list[PageText] = []
        empty = 0
        for index in range(total):
            try:
                raw = doc.load_page(index).get_text("text")
            except Exception as exc:
                raise CorruptedPDFError(f"Page {index + 1} could not be read; the PDF may be damaged.") from exc
            text = clean_text(raw)
            if len(text) < min_chars:
                empty += 1
                continue
            pages.append(PageText(page=index + 1, text=text))
    finally:
        doc.close()

    if not pages:
        raise NoExtractableTextError(
            "No extractable text was found. This looks like a scanned or image-only PDF. "
            "DocuMind AI does not perform OCR, so run OCR first (for example with OCRmyPDF) and re-upload."
        )
    return ExtractedDocument(pages=pages, total_pages=total, empty_pages=empty)
