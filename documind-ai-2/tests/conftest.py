"""Shared fixtures: synthetic PDFs, fake LLM backends and wired services (hashing embedder)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from config import Settings  # noqa: E402
from src.container import build_services  # noqa: E402
from src.embeddings import HashingEmbedder  # noqa: E402
from src.security import new_session_id  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def make_pdf(pages: list[str], password: str | None = None) -> bytes:
    """Build a small text PDF; each string becomes one page ('' gives a blank page)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    if password:
        data = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
        )
    else:
        data = doc.tobytes()
    doc.close()
    return data


PHOTO_TEXT = (
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
    "Chlorophyll inside the chloroplasts absorbs sunlight, mainly red and blue wavelengths. "
    "The light reactions produce ATP and NADPH, while the Calvin cycle fixes carbon dioxide into sugars."
)
ROMAN_TEXT = (
    "The Roman Republic was founded in 509 BC after the overthrow of the last king. "
    "Two consuls were elected each year to lead the government and the army. "
    "The Senate advised the consuls and controlled public finances."
)
SPACE_TEXT = (
    "Jupiter is the largest planet in the solar system with a mass over twice that of all other planets combined. "
    "Its Great Red Spot is a giant storm that has been observed for centuries. "
    "Jupiter has at least ninety known moons including Ganymede, the largest moon in the solar system."
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gemini_api_key="test-key-not-real",
        embedding_backend="hashing",
        chunk_size=400,
        chunk_overlap=60,
        top_k=4,
        fetch_k=12,
        min_similarity=0.12,
        questions_per_window=100,
    )


class FakeLLM:
    """Records calls and returns a scripted reply (a str or a callable(system, prompt) -> str)."""

    def __init__(self, reply="", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def generate(self, system, prompt, json_output=True, max_output_tokens=None):
        self.calls.append({"system": system, "prompt": prompt})
        if self.error:
            raise self.error
        return self.reply(system, prompt) if callable(self.reply) else self.reply


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings, fake_llm):
    return build_services(settings, llm=fake_llm, embedder=HashingEmbedder(), store=VectorStore())


@pytest.fixture
def session_id() -> str:
    return new_session_id()


@pytest.fixture
def science_pdf() -> bytes:
    return make_pdf([PHOTO_TEXT, SPACE_TEXT])


@pytest.fixture
def history_pdf() -> bytes:
    return make_pdf([ROMAN_TEXT])
