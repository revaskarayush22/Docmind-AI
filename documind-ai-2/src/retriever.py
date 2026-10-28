"""Retrieval: semantic search + optional BM25 hybrid fusion, optional reranking,
evidence threshold, near-duplicate removal and a per-page diversity cap."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from config import Settings

from .embeddings import Embedder
from .vector_store import StoredChunk, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_K = 60


@dataclass
class RetrievalResult:
    chunks: list[StoredChunk]  # final, ordered by relevance
    best_similarity: float  # best *semantic* cosine similarity seen
    sufficient: bool  # does the evidence clear the configured threshold?
    reason: str = ""
    debug: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Small self-contained Okapi BM25 (avoids a hard dependency for the core path)."""
    q_terms = set(tokenize(query))
    tokenized = [tokenize(d) for d in docs]
    n = len(tokenized)
    if n == 0 or not q_terms:
        return [0.0] * n
    avgdl = sum(len(t) for t in tokenized) / n or 1.0
    df = {term: sum(1 for t in tokenized if term in t) for term in q_terms}
    scores = []
    for toks in tokenized:
        tf: dict[str, int] = {}
        for tok in toks:
            if tok in q_terms:
                tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(toks) / avgdl))
        scores.append(score)
    return scores


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        reranker: Optional[Callable[[str, list[StoredChunk]], list[float]]] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._reranker = reranker
        self._reranker_loaded = reranker is not None

    # -- optional reranker ---------------------------------------------------
    def _get_reranker(self):
        if self._reranker_loaded:
            return self._reranker
        self._reranker_loaded = True
        if not self.settings.use_reranker:
            return None
        try:  # pragma: no cover - needs model download
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.settings.reranker_model, device="cpu")
            self._reranker = lambda q, chunks: [float(s) for s in model.predict([(q, c.text) for c in chunks])]
        except Exception:
            self._reranker = None  # degrade gracefully to un-reranked results
        return self._reranker

    # -- main entry ----------------------------------------------------------
    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[list[str]] = None,
        min_similarity: Optional[float] = None,
    ) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.top_k
        threshold = s.min_similarity if min_similarity is None else min_similarity
        q_vec = self.embedder.embed([query])[0]
        semantic = self.store.query(session_id, q_vec, max(s.fetch_k, top_k), doc_ids)
        if not semantic:
            return RetrievalResult([], 0.0, False, "No documents are indexed for this session.")

        best_sim = max(c.score for c in semantic)
        sim_by_id = {c.chunk_id: c.score for c in semantic}
        candidates: dict[str, StoredChunk] = {c.chunk_id: c for c in semantic}
        fused: dict[str, float] = {}
        for rank, c in enumerate(semantic):
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        lexical_hits: set[str] = set()
        if s.use_hybrid:
            corpus = self.store.all_chunks(session_id, doc_ids)
            scores = bm25_scores(query, [c.text for c in corpus])
            ranked = sorted(
                ((sc, c) for sc, c in zip(scores, corpus) if sc > 0), key=lambda x: x[0], reverse=True
            )[: s.fetch_k]
            for rank, (_, c) in enumerate(ranked):
                lexical_hits.add(c.chunk_id)
                candidates.setdefault(c.chunk_id, c)
                fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        ordered = sorted(candidates.values(), key=lambda c: fused[c.chunk_id], reverse=True)

        reranker = self._get_reranker()
        if reranker is not None and ordered:
            head = ordered[: s.fetch_k]
            scores = reranker(query, head)
            ordered = [c for _, c in sorted(zip(scores, head), key=lambda x: x[0], reverse=True)]

        # Keep chunks that carry real evidence: semantic similarity above threshold, or a
        # lexical (BM25) hit when hybrid is on. Pure-semantic junk below threshold is dropped.
        def passes(c: StoredChunk) -> bool:
            sim = sim_by_id.get(c.chunk_id)
            return (sim is not None and sim >= threshold) or c.chunk_id in lexical_hits

        ordered = [c for c in ordered if passes(c)]

        deduped = self._dedupe(ordered)
        diverse = self._cap_per_page(deduped, top_k)
        final = diverse[:top_k]

        sufficient = bool(final) and best_sim >= threshold
        reason = "" if sufficient else (
            f"No passage was similar enough to the question (best similarity {best_sim:.2f}, "
            f"threshold {threshold:.2f})."
        )
        # Re-attach semantic scores to chunks that only matched by keyword (score stays 0).
        return RetrievalResult(
            chunks=final,
            best_similarity=best_sim,
            sufficient=sufficient,
            reason=reason,
            debug={"candidates": len(candidates), "after_dedupe": len(deduped)},
        )

    def _dedupe(self, chunks: list[StoredChunk]) -> list[StoredChunk]:
        kept: list[StoredChunk] = []
        for c in chunks:
            if all(jaccard(c.text, k.text) < self.settings.dedupe_threshold for k in kept):
                kept.append(c)
        return kept

    def _cap_per_page(self, chunks: list[StoredChunk], top_k: int) -> list[StoredChunk]:
        """Limit chunks per (doc, page) so multi-page questions see several pages.
        Overflow chunks are only used if there are not enough others to fill top_k."""
        cap = max(1, self.settings.max_chunks_per_page)
        counts: dict[tuple[str, int], int] = {}
        primary, overflow = [], []
        for c in chunks:
            key = (c.doc_id, c.page)
            if counts.get(key, 0) < cap:
                counts[key] = counts.get(key, 0) + 1
                primary.append(c)
            else:
                overflow.append(c)
        return primary + overflow if len(primary) < top_k else primary


def order_for_reading(chunks: list[StoredChunk]) -> list[StoredChunk]:
    """Reading order (document, page, position) for building the LLM context."""
    return sorted(chunks, key=lambda c: (c.filename, c.page, c.index))
