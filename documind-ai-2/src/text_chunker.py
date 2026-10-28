"""Page-aware chunking with overlap.

Chunks never span pages, so every chunk maps to exactly one page number and
citations can be exact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_processor import PageText


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    index: int  # position of the chunk within the whole document
    text: str
    metadata: dict = field(default_factory=dict)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split_units(text: str) -> list[str]:
    """Split into sentence-like units, keeping paragraph boundaries as separate units."""
    units: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        units.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return units


def _hard_wrap(unit: str, size: int) -> list[str]:
    """Break an over-long unit (no sentence punctuation) on whitespace."""
    out, current = [], ""
    for word in unit.split():
        if len(word) > size:  # pathological token: cut it
            if current:
                out.append(current)
                current = ""
            out.extend(word[i : i + size] for i in range(0, len(word), size))
            continue
        if current and len(current) + 1 + len(word) > size:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Chunks are built from whole sentences where possible. Each new chunk starts with
    trailing sentences of the previous chunk totalling at most ``overlap`` characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    units: list[str] = []
    for unit in _split_units(text):
        units.extend(_hard_wrap(unit, chunk_size) if len(unit) > chunk_size else [unit])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append(" ".join(current))
            # carry over an overlap tail made of whole trailing units
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                extra = len(prev) + (1 if tail else 0)
                if tail_len + extra > overlap:
                    break
                tail.insert(0, prev)
                tail_len += extra
            current, length = tail, tail_len
            added = len(unit) + (1 if current else 0)
            if length + added > chunk_size:  # overlap tail + unit too big: drop the tail
                current, length, added = [], 0, len(unit)
        current.append(unit)
        length += added
    if current:
        last = " ".join(current)
        # avoid emitting a final chunk that is only a repeat of the previous overlap
        if not chunks or last not in chunks[-1]:
            chunks.append(last)
    return chunks


def chunk_pages(
    pages: list[PageText], doc_id: str, filename: str, chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{index:05d}",
                    doc_id=doc_id,
                    filename=filename,
                    page=page.page,
                    index=index,
                    text=piece,
                )
            )
    return chunks
