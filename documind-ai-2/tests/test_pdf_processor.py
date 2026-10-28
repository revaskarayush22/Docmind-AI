import pytest

from src.pdf_processor import (
    CorruptedPDFError,
    EncryptedPDFError,
    NoExtractableTextError,
    TooManyPagesError,
    clean_text,
    extract_pages,
)
from tests.conftest import PHOTO_TEXT, SPACE_TEXT, make_pdf


def test_extracts_text_with_page_numbers():
    doc = extract_pages(make_pdf([PHOTO_TEXT, SPACE_TEXT]))
    assert doc.total_pages == 2
    assert [p.page for p in doc.pages] == [1, 2]
    assert "Photosynthesis" in doc.pages[0].text
    assert "Jupiter" in doc.pages[1].text


def test_page_numbers_skip_blank_pages_but_stay_correct():
    doc = extract_pages(make_pdf([PHOTO_TEXT, "", SPACE_TEXT]))
    assert [p.page for p in doc.pages] == [1, 3]  # blank page 2 skipped, numbering preserved
    assert doc.empty_pages == 1 and doc.total_pages == 3


def test_image_only_or_blank_pdf_is_reported_not_processed():
    with pytest.raises(NoExtractableTextError) as exc:
        extract_pages(make_pdf(["", ""]))
    assert "OCR" in str(exc.value)


def test_corrupted_bytes_raise_friendly_error():
    with pytest.raises(CorruptedPDFError):
        extract_pages(b"%PDF-1.7 this is definitely not a valid pdf body")


def test_truncated_pdf_does_not_crash_with_unhandled_error():
    data = make_pdf([PHOTO_TEXT])
    with pytest.raises((CorruptedPDFError, NoExtractableTextError)):
        extract_pages(data[: len(data) // 3])


def test_encrypted_pdf_is_rejected():
    with pytest.raises(EncryptedPDFError):
        extract_pages(make_pdf([PHOTO_TEXT], password="secret"))


def test_page_limit_enforced():
    with pytest.raises(TooManyPagesError):
        extract_pages(make_pdf([PHOTO_TEXT] * 4), max_pages=3)


def test_clean_text_rejoins_hyphenation_and_collapses_whitespace():
    raw = "inter-\nnational   trade\nis  growing.\n\n\n\nNew paragraph\x00 here."
    cleaned = clean_text(raw)
    assert "international trade is growing." in cleaned
    assert "\n\n\n" not in cleaned
    assert "\x00" not in cleaned
    assert "New paragraph here." in cleaned
