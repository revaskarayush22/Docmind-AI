import json

import pytest

from src.llm_service import LLMResponseError
from src.study_tools import StudyResult, StudyItem, build_study_guide_markdown, sample_chunks
from src.vector_store import StoredChunk


@pytest.fixture
def loaded(services, session_id, science_pdf):
    services.manager.add_document(session_id, "science.pdf", science_pdf)
    return services, session_id


def test_quiz_items_are_validated_and_pages_mapped(loaded, fake_llm):
    services, sid = loaded
    fake_llm.reply = json.dumps(
        {
            "items": [
                {"question": "What absorbs sunlight?", "options": ["Chlorophyll", "Water", "Soil", "Wind"], "answer_index": 0, "explanation": "Chlorophyll absorbs light.", "sources": [1, 99]},
                {"question": "Broken?", "options": ["a"], "answer_index": 5, "explanation": "", "sources": [1]},
            ]
        }
    )
    res = services.study.generate(sid, "quiz")
    assert len(res.items) == 1  # malformed item dropped
    assert res.items[0].pages and res.items[0].pages[0].startswith("science.pdf p. ")
    assert any("invalid" in w for w in res.warnings)  # fabricated source 99 dropped


def test_summary_faq_key_points(loaded, fake_llm):
    services, sid = loaded
    fake_llm.reply = json.dumps({"summary": "Plants convert light into energy.", "sources": [1]})
    assert services.study.generate(sid, "summary").items[0].text.startswith("Plants")
    fake_llm.reply = json.dumps({"items": [{"point": "Chlorophyll absorbs light", "sources": [1]}]})
    assert services.study.generate(sid, "key_points").items[0].pages
    fake_llm.reply = json.dumps({"items": [{"question": "What is the Calvin cycle?", "answer": "It fixes CO2.", "sources": [1]}]})
    assert services.study.generate(sid, "faq").items[0].extra["answer"] == "It fixes CO2."


def test_study_failure_raises_but_chat_unaffected(loaded, fake_llm):
    services, sid = loaded
    fake_llm.reply = "not json"
    with pytest.raises(LLMResponseError):
        services.study.generate(sid, "summary")
    fake_llm.reply = json.dumps({"status": "answered", "answer": "Chlorophyll absorbs sunlight [S1].", "limitations": ""})
    assert services.pipeline.ask(sid, "What does chlorophyll absorb?").citations


def test_no_documents_makes_no_llm_call(services, session_id, fake_llm):
    res = services.study.generate(session_id, "summary")
    assert res.items == [] and fake_llm.calls == []


def test_sampling_spreads_across_document_and_respects_budget():
    chunks = [StoredChunk(f"c{i}", "d", "f.pdf", i + 1, i, "word " * 100) for i in range(100)]
    picked = sample_chunks(chunks, max_chars=5000)
    assert sum(len(c.text) for c in picked) <= 5000 + 500
    assert picked[0].page < 10 and picked[-1].page > 80


def test_study_guide_markdown_contains_sections_and_answer_key():
    results = {
        "summary": StudyResult("summary", [StudyItem("A summary.", ["f.pdf p. 1"])]),
        "quiz": StudyResult("quiz", [StudyItem("Q?", ["f.pdf p. 2"], {"options": ["x", "y"], "answer_index": 1, "explanation": "because"})]),
    }
    md = build_study_guide_markdown("f.pdf", results)
    assert "## Summary" in md and "### Answer key" in md and "**B**" in md and "verify" in md.lower()
