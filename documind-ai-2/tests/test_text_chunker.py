import pytest

from src.pdf_processor import PageText
from src.text_chunker import chunk_pages, split_text

SENTENCES = " ".join(f"This is sentence number {i} about topic {i % 7}." for i in range(60))


def test_chunks_respect_max_size():
    chunks = split_text(SENTENCES, chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(SENTENCES, chunk_size=300, overlap=80)
    overlaps = 0
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.split(". ")[-1].rstrip(".")
        if last_sentence and last_sentence in b:
            overlaps += 1
    assert overlaps >= len(chunks) - 2  # essentially every boundary shares trailing text


def test_no_overlap_when_zero():
    chunks = split_text(SENTENCES, chunk_size=300, overlap=0)
    joined = " ".join(chunks)
    assert joined.count("sentence number 10 ") == 1


def test_all_content_is_covered():
    chunks = split_text(SENTENCES, chunk_size=300, overlap=60)
    joined = " ".join(chunks)
    for i in range(60):
        assert f"sentence number {i} " in joined


def test_short_text_is_single_chunk_and_empty_is_none():
    assert split_text("Short text.", 300, 50) == ["Short text."]
    assert split_text("   ", 300, 50) == []


def test_overlong_unbroken_token_is_hard_wrapped():
    import hashlib

    token = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(16))[:1000]  # non-repeating
    chunks = split_text(token, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks) and len(chunks) >= 4
    assert "".join(chunks) == token  # hard wrap loses nothing


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("hello world", size, overlap)


def test_chunk_metadata_and_page_boundaries():
    pages = [PageText(1, SENTENCES[:700]), PageText(4, "Page four text is here and long enough to keep.")]
    chunks = chunk_pages(pages, "dabc", "file.pdf", chunk_size=300, overlap=50)
    assert {c.page for c in chunks} == {1, 4}
    assert all(c.doc_id == "dabc" and c.filename == "file.pdf" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # chunks never mix pages: the page-4 text only appears in page-4 chunks
    assert all("Page four" not in c.text for c in chunks if c.page == 1)


def test_empty_document_yields_no_chunks():
    assert chunk_pages([], "d1", "f.pdf") == []
