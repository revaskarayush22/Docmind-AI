"""Headless UI smoke tests using Streamlit's AppTest (no browser, no real API calls)."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.container as container
from src.container import build_services
from src.embeddings import HashingEmbedder
from src.security import new_session_id
from src.vector_store import VectorStore
from tests.conftest import FakeLLM, ROMAN_TEXT, make_pdf

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def wired(monkeypatch, settings):
    llm = FakeLLM()
    svc = build_services(settings, llm=llm, embedder=HashingEmbedder(), store=VectorStore())
    monkeypatch.setattr(container, "build_services", lambda s: svc)
    import streamlit as st

    st.cache_resource.clear()
    return svc, llm


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
    parts += [i.value for i in at.info] + [c.value for c in at.caption]
    return "\n".join(parts)


def test_landing_page_renders_without_exceptions(wired):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = _all_text(at)
    assert "DocuMind AI" in text and "Upload a PDF" in text
    assert "Privacy notice" in " ".join(e.label for e in at.sidebar.expander)


def test_missing_api_key_banner(wired, monkeypatch):
    svc, _ = wired
    object.__setattr__(svc.settings, "gemini_api_key", "")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("GEMINI_API_KEY" in e.value for e in at.error)


def test_question_flow_shows_answer_and_sources(wired):
    svc, llm = wired
    sid = new_session_id()
    assert svc.manager.add_document(sid, "rome.pdf", make_pdf([ROMAN_TEXT])).ok
    llm.reply = json.dumps({"status": "answered", "answer": "Two consuls were elected each year [S1].", "limitations": ""})
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["session_id"] = sid
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("How many consuls were elected each year?").run()
    assert not at.exception
    text = _all_text(at) + " ".join(e.label for e in at.expander)
    assert "Two consuls were elected each year" in text
    assert "rome.pdf" in text and "Sources (1)" in text


def test_llm_failure_shows_friendly_error_not_traceback(wired):
    from src.llm_service import LLMRateLimitError

    svc, llm = wired
    sid = new_session_id()
    svc.manager.add_document(sid, "rome.pdf", make_pdf([ROMAN_TEXT]))
    llm.error = LLMRateLimitError("The Gemini API quota or rate limit was reached.")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["session_id"] = sid
    at.run()
    at.chat_input[0].set_value("How many consuls were elected each year?").run()
    assert not at.exception
    assert any("quota" in e.value for e in at.error)


def test_clear_session_removes_documents(wired):
    svc, _ = wired
    sid = new_session_id()
    svc.manager.add_document(sid, "rome.pdf", make_pdf([ROMAN_TEXT]))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["session_id"] = sid
    at.run()
    next(b for b in at.sidebar.button if b.label == "Clear session").click().run()
    assert not at.exception
    assert svc.store.count(sid) == 0
    assert at.session_state["session_id"] != sid
