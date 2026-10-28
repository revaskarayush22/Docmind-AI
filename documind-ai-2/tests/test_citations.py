from src.citations import (
    LOW_SUPPORT,
    build_cited_answer,
    best_excerpt,
    format_citation,
    normalize_markers,
    support_score,
)
from src.vector_store import StoredChunk


def chunk(n, text, page=1, name="doc.pdf"):
    return StoredChunk(f"c{n}", "d1", name, page, n, text)


RETRIEVED = [
    chunk(0, "The Roman Republic was founded in 509 BC. Two consuls were elected each year.", page=3, name="rome.pdf"),
    chunk(1, "The Senate advised the consuls and controlled public finances.", page=4, name="rome.pdf"),
]


def test_group_markers_are_expanded():
    assert normalize_markers("Fact [S1, S2].") == "Fact [S1][S2]."


def test_valid_citations_map_to_filename_page_and_excerpt():
    ans = build_cited_answer("Two consuls were elected each year [S1]. The Senate advised them [S2].", RETRIEVED)
    assert [c.number for c in ans.citations] == [1, 2]
    assert ans.citations[0].filename == "rome.pdf" and ans.citations[0].page == 3
    assert "consuls" in ans.citations[0].excerpt
    assert ans.invalid_markers == []
    assert ans.segments[0].source_numbers == [1] and ans.segments[1].source_numbers == [2]


def test_citation_to_non_retrieved_chunk_is_dropped():
    ans = build_cited_answer("Consuls were elected annually [S1]. The empire fell in 476 AD [S9].", RETRIEVED)
    assert ans.invalid_markers == [9]
    assert [c.number for c in ans.citations] == [1]
    assert "[S9]" not in ans.text


def test_answer_without_markers_has_no_citations():
    ans = build_cited_answer("Consuls were elected annually.", RETRIEVED)
    assert ans.citations == [] and ans.segments[0].support is None


def test_support_score_high_for_grounded_and_low_for_unrelated():
    passages = [RETRIEVED[0].text]
    grounded = support_score("Two consuls were elected each year", passages)
    ungrounded = support_score("Volcanic eruptions destroyed Pompeii overnight", passages)
    assert grounded > 0.8 and ungrounded < LOW_SUPPORT


def test_weakly_supported_segment_flagged():
    ans = build_cited_answer("Volcanic eruptions destroyed Pompeii overnight [S1].", RETRIEVED)
    assert ans.segments[0].weakly_supported


def test_best_excerpt_picks_the_relevant_sentence_and_truncates():
    passage = "Unrelated opening sentence. The Senate controlled public finances. Another closing sentence."
    assert best_excerpt(passage, "who controlled finances") == "The Senate controlled public finances."
    long = "word " * 200
    assert len(best_excerpt(long, "word")) <= 280


def test_format_citation_contains_file_page_and_excerpt():
    ans = build_cited_answer("Consuls were elected each year [S1].", RETRIEVED)
    text = format_citation(ans.citations[0])
    assert "rome.pdf" in text and "page 3" in text and "[S1]" in text


def test_marker_at_start_of_segment_attaches_to_previous_sentence():
    ans = build_cited_answer("Consuls were elected each year. [S1] The Senate advised them [S2].", RETRIEVED)
    assert ans.segments[0].source_numbers == [1]
