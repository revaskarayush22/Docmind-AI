"""Citation parsing, validation and display helpers.

The model is told to cite with ``[S1]``-style markers that refer to the numbered
passages it was given. This module:

* rejects markers that do not correspond to a passage that was actually retrieved,
* splits the answer into segments and records which sources each one cites,
* picks a short supporting excerpt from the cited passage, and
* computes a lexical support score as a cheap sanity signal (NOT a proof of
  faithfulness - it only measures word overlap).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .vector_store import StoredChunk

_MARKER_RE = re.compile(r"\[S(\d+)\]")
_GROUP_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)+)\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the and or of to in on for with is are was were be been it this that as at by from".split())

LOW_SUPPORT = 0.25  # below this a cited sentence is flagged as weakly supported


@dataclass(frozen=True)
class Citation:
    number: int  # the S-number shown to the model and the user
    chunk_id: str
    filename: str
    page: int
    excerpt: str

    @property
    def label(self) -> str:
        return f"{self.filename}, p. {self.page}"


@dataclass
class Segment:
    text: str
    source_numbers: list[int] = field(default_factory=list)
    support: float | None = None  # lexical overlap with cited passages (None if uncited)

    @property
    def weakly_supported(self) -> bool:
        return self.support is not None and self.support < LOW_SUPPORT


@dataclass
class CitedAnswer:
    text: str  # answer with markers normalised to [S#], invalid ones removed
    segments: list[Segment]
    citations: list[Citation]  # only those actually cited, ordered by number
    invalid_markers: list[int]  # numbers the model cited that were not retrieved


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def normalize_markers(text: str) -> str:
    """Turn ``[S1, S2]`` into ``[S1][S2]``."""

    def expand(m: re.Match) -> str:
        return "".join(f"[{p.strip()}]" for p in re.split(r"[,;]", m.group(1)))

    return _GROUP_RE.sub(expand, text)


def support_score(sentence: str, passages: list[str]) -> float:
    """Fraction of the sentence's content words that appear in the cited passages."""
    sent = _words(_MARKER_RE.sub("", sentence))
    if not sent:
        return 1.0
    evidence: set[str] = set()
    for p in passages:
        evidence |= _words(p)
    return len(sent & evidence) / len(sent)


def best_excerpt(passage: str, claim: str, max_chars: int = 280) -> str:
    """Return the passage sentence(s) that best overlap with ``claim``, truncated."""
    sentences = [s.strip() for s in _SENT_RE.split(passage) if s.strip()] or [passage.strip()]
    claim_words = _words(claim)
    scored = sorted(
        range(len(sentences)),
        key=lambda i: (-len(claim_words & _words(sentences[i])), i),
    )
    best = sentences[scored[0]]
    if len(best) > max_chars:
        best = best[: max_chars - 1].rstrip() + "…"
    return best


def build_cited_answer(answer: str, retrieved: list[StoredChunk]) -> CitedAnswer:
    """Validate markers against ``retrieved`` (1-based S-numbers) and build segments."""
    answer = normalize_markers(answer.strip())
    by_number = {i + 1: c for i, c in enumerate(retrieved)}

    invalid: list[int] = []

    def drop_invalid(m: re.Match) -> str:
        n = int(m.group(1))
        if n in by_number:
            return m.group(0)
        if n not in invalid:
            invalid.append(n)
        return ""

    cleaned = _MARKER_RE.sub(drop_invalid, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)

    segments: list[Segment] = []
    cited_numbers: set[int] = set()
    for raw in re.split(r"(?<=[.!?])\s+(?=\S)|\n+", cleaned):
        raw = raw.strip()
        if not raw:
            continue
        # A marker group that landed at the start of a segment belongs to the previous sentence.
        lead = re.match(r"^((?:\[S\d+\])+)\s*(.*)$", raw)
        if lead and segments:
            nums = [int(n) for n in _MARKER_RE.findall(lead.group(1))]
            for n in nums:
                if n not in segments[-1].source_numbers:
                    segments[-1].source_numbers.append(n)
            segments[-1].text += lead.group(1)
            raw = lead.group(2).strip()
            if not raw:
                continue
        nums = []
        for n in (int(x) for x in _MARKER_RE.findall(raw)):
            if n not in nums:
                nums.append(n)
        segments.append(Segment(text=raw, source_numbers=nums))

    for seg in segments:
        if seg.source_numbers:
            cited_numbers.update(seg.source_numbers)
            seg.support = support_score(seg.text, [by_number[n].text for n in seg.source_numbers])

    citations = []
    for n in sorted(cited_numbers):
        chunk = by_number[n]
        claim = " ".join(seg.text for seg in segments if n in seg.source_numbers)
        citations.append(
            Citation(
                number=n,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page=chunk.page,
                excerpt=best_excerpt(chunk.text, _MARKER_RE.sub("", claim)),
            )
        )
    return CitedAnswer(text=cleaned, segments=segments, citations=citations, invalid_markers=invalid)


def format_citation(c: Citation) -> str:
    return f"[S{c.number}] {c.filename}, page {c.page}: “{c.excerpt}”"


def format_sources_markdown(citations: list[Citation]) -> str:
    return "\n".join(f"- {format_citation(c)}" for c in citations)
