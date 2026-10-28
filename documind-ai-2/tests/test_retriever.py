from src.embeddings import HashingEmbedder
from src.retriever import Retriever, bm25_scores, jaccard
from src.security import new_session_id
from src.vector_store import VectorStore
from tests.conftest import make_pdf


def _ingest(services, session_id, name, data):
    outcome = services.manager.add_document(session_id, name, data)
    assert outcome.ok, outcome.message
    return outcome.record


def test_relevant_chunk_is_retrieved_first(services, session_id, science_pdf, history_pdf):
    _ingest(services, session_id, "science.pdf", science_pdf)
    _ingest(services, session_id, "history.pdf", history_pdf)
    res = services.retriever.retrieve(session_id, "How many consuls were elected in the Roman Republic?")
    assert res.sufficient
    assert res.chunks[0].filename == "history.pdf"
    res = services.retriever.retrieve(session_id, "What does chlorophyll absorb?")
    assert res.chunks[0].filename == "science.pdf" and res.chunks[0].page == 1


def test_metadata_filter_restricts_to_selected_documents(services, session_id, science_pdf, history_pdf):
    _ingest(services, session_id, "science.pdf", science_pdf)
    hist = _ingest(services, session_id, "history.pdf", history_pdf)
    res = services.retriever.retrieve(session_id, "Jupiter largest planet", doc_ids=[hist.doc_id])
    assert all(c.doc_id == hist.doc_id for c in res.chunks)
    empty = services.retriever.retrieve(session_id, "Jupiter", doc_ids=[])
    assert empty.chunks == [] and not empty.sufficient


def test_sessions_are_isolated(services, science_pdf, history_pdf):
    a, b = new_session_id(), new_session_id()
    _ingest(services, a, "science.pdf", science_pdf)
    _ingest(services, b, "history.pdf", history_pdf)
    res_b = services.retriever.retrieve(b, "Jupiter largest planet moons")
    assert all(c.filename == "history.pdf" for c in res_b.chunks)
    res_a = services.retriever.retrieve(a, "Roman consuls Senate")
    assert all(c.filename == "science.pdf" for c in res_a.chunks)
    assert services.store.count(a) > 0 and services.store.count(b) > 0


def test_unknown_session_returns_nothing(services):
    res = services.retriever.retrieve(new_session_id(), "anything at all")
    assert res.chunks == [] and not res.sufficient


def test_irrelevant_question_is_not_sufficient(settings, services, session_id, science_pdf):
    _ingest(services, session_id, "science.pdf", science_pdf)
    strict = settings.with_overrides(min_similarity=0.6, use_hybrid=False)
    r = Retriever(services.store, services.embedder, strict)
    res = r.retrieve(session_id, "Who won the football world cup in 1998?")
    assert not res.sufficient and res.reason


def test_bm25_prefers_exact_keyword_matches():
    docs = ["apples and oranges", "the ganymede moon is large", "nothing relevant"]
    scores = bm25_scores("ganymede moon", docs)
    assert scores[1] > scores[0] and scores[1] > scores[2]


def test_hybrid_finds_keyword_hit_pure_semantic_may_rank_lower(settings, services, session_id):
    text = "Introduction. " + "Filler sentence about nothing in particular. " * 8 + "The serial code ZX-4417 identifies the unit."
    _ingest(services, session_id, "manual.pdf", make_pdf([text]))
    hybrid = Retriever(services.store, services.embedder, settings.with_overrides(use_hybrid=True, min_similarity=0.0))
    res = hybrid.retrieve(session_id, "ZX-4417")
    assert any("ZX-4417" in c.text for c in res.chunks)


def test_near_duplicates_are_removed(settings, services, session_id):
    page = "The warranty lasts twelve months from the date of purchase and covers manufacturing defects."
    _ingest(services, session_id, "a.pdf", make_pdf([page]))
    _ingest(services, session_id, "b.pdf", make_pdf([page + " "]))  # same text, different file bytes
    r = Retriever(services.store, services.embedder, settings.with_overrides(min_similarity=0.0))
    res = r.retrieve(session_id, "warranty twelve months")
    texts = [c.text for c in res.chunks]
    assert len(texts) == len(set(texts)) and len(res.chunks) == 1


def test_per_page_cap_promotes_multi_page_evidence(settings, services, session_id):
    p1 = " ".join(f"Budget item {i} costs money for the project plan." for i in range(30))
    p2 = "The project budget total appears on this second page together with the plan costs."
    _ingest(services, session_id, "plan.pdf", make_pdf([p1[:1500], p2]))
    r = Retriever(services.store, services.embedder, settings.with_overrides(max_chunks_per_page=1, min_similarity=0.0, top_k=2))
    res = r.retrieve(session_id, "project budget plan costs")
    assert {c.page for c in res.chunks} == {1, 2}


def test_reranker_can_reorder_results(settings, services, session_id, science_pdf):
    _ingest(services, session_id, "science.pdf", science_pdf)
    base = services.retriever.retrieve(session_id, "planet moons storm chlorophyll")
    assert len(base.chunks) >= 2
    reverse = Retriever(
        services.store,
        services.embedder,
        settings,
        reranker=lambda q, chunks: [float(i) for i in range(len(chunks))],  # last candidate ranked first
    )
    reranked = reverse.retrieve(session_id, "planet moons storm chlorophyll")
    assert reranked.chunks[0].chunk_id != base.chunks[0].chunk_id


def test_jaccard_bounds():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b", "c d") == 0.0
    assert jaccard("", "x") == 0.0
