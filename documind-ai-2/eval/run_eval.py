"""Reproducible evaluation for DocuMind AI.

Offline mode (default, no API key needed) measures RETRIEVAL only:
  Recall@k, Mean Reciprocal Rank, multi-page coverage, abstention behaviour (evidence
  threshold), and retrieval latency, with sweeps over chunk size / hybrid / top-k / threshold.

Live mode (--live, needs GEMINI_API_KEY) additionally runs the full pipeline and measures:
  citation correctness, answer relevance (keyword hit rate), a lexical faithfulness proxy,
  correct abstention, latency and estimated token usage.

Usage:
  python -m eval.run_eval --embedder hashing
  python -m eval.run_eval --embedder sentence-transformers          # production model
  python -m eval.run_eval --embedder sentence-transformers --live   # + answer-quality metrics
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from config import Settings  # noqa: E402
from src.container import build_services  # noqa: E402
from src.embeddings import get_embedder  # noqa: E402
from src.rag_pipeline import AnswerStatus  # noqa: E402
from src.security import new_session_id  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

DATASET = Path(__file__).with_name("dataset.json")
RESULTS_DIR = Path(__file__).with_name("results")


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def build_session(settings: Settings, embedder, dataset: dict, llm=None):
    svc = build_services(settings, llm=llm, embedder=embedder, store=VectorStore())
    sid = new_session_id()
    t0 = time.perf_counter()
    for name, pages in dataset["documents"].items():
        out = svc.manager.add_document(sid, name, make_pdf(pages))
        assert out.ok, out.message
    return svc, sid, time.perf_counter() - t0


def is_hit(chunk, relevant) -> bool:
    return any(chunk.filename == d and chunk.page == p for d, p in relevant)


def retrieval_metrics(settings: Settings, embedder, dataset: dict, ks=(1, 3, 5)) -> dict:
    """Quality of ranking with the evidence gate disabled (gate is evaluated separately)."""
    cfg = settings.with_overrides(min_similarity=-1.0, top_k=max(ks), fetch_k=max(20, max(ks)), max_chunks_per_page=max(ks))
    svc, sid, ingest_s = build_session(cfg, embedder, dataset)
    answerable = [q for q in dataset["questions"] if q["relevant"]]
    recalls = {k: [] for k in ks}
    rr, coverage, latencies = [], [], []
    sims = {}
    for q in dataset["questions"]:
        t0 = time.perf_counter()
        res = svc.retriever.retrieve(sid, q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        sims[q["id"]] = res.best_similarity
        if not q["relevant"]:
            continue
        ranked = res.chunks
        for k in ks:
            recalls[k].append(any(is_hit(c, q["relevant"]) for c in ranked[:k]))
        first = next((i for i, c in enumerate(ranked, 1) if is_hit(c, q["relevant"])), None)
        rr.append(1.0 / first if first else 0.0)
        if len(q["relevant"]) > 1:
            found = {(c.filename, c.page) for c in ranked[: max(ks)]}
            coverage.append(sum((d, p) in found for d, p in q["relevant"]) / len(q["relevant"]))
    by_type: dict[str, list[bool]] = {}
    for q in answerable:
        ranked = svc.retriever.retrieve(sid, q["question"]).chunks
        by_type.setdefault(q["type"], []).append(any(is_hit(c, q["relevant"]) for c in ranked[:3]))
    return {
        "n_answerable": len(answerable),
        "recall@k": {f"@{k}": round(sum(v) / len(v), 3) for k, v in recalls.items()},
        "mrr": round(sum(rr) / len(rr), 3),
        "multi_page_coverage@5": round(sum(coverage) / len(coverage), 3) if coverage else None,
        "recall@3_by_type": {t: round(sum(v) / len(v), 3) for t, v in by_type.items()},
        "retrieval_latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        },
        "ingest_seconds": round(ingest_s, 2),
        "best_similarity": sims,
    }


def abstention_table(best_sims: dict[str, float], dataset: dict, thresholds) -> list[dict]:
    """How the evidence threshold trades missed answers against refusing unanswerable questions."""
    rows = []
    ans = [q for q in dataset["questions"] if q["relevant"]]
    unans = [q for q in dataset["questions"] if not q["relevant"]]
    for t in thresholds:
        false_abstain = sum(best_sims[q["id"]] < t for q in ans) / len(ans)
        correct_abstain = sum(best_sims[q["id"]] < t for q in unans) / len(unans)
        rows.append({"threshold": t, "false_abstain_rate": round(false_abstain, 3), "correct_abstain_rate": round(correct_abstain, 3)})
    return rows


def run_offline(embedder, embedder_label: str, dataset: dict) -> dict:
    base = Settings(embedding_backend="hashing", gemini_api_key="")
    out: dict = {"embedder": embedder_label, "is_semantic": embedder.is_semantic}
    out["baseline"] = retrieval_metrics(base, embedder, dataset)
    sweeps = {"chunk_size": [], "hybrid": [], "dedupe_and_page_cap": []}
    for size in (300, 600, 900, 1500):
        m = retrieval_metrics(base.with_overrides(chunk_size=size, chunk_overlap=size // 6), embedder, dataset)
        sweeps["chunk_size"].append({"chunk_size": size, "recall@1": m["recall@k"]["@1"], "recall@3": m["recall@k"]["@3"], "mrr": m["mrr"]})
    for hybrid in (False, True):
        m = retrieval_metrics(base.with_overrides(use_hybrid=hybrid), embedder, dataset)
        sweeps["hybrid"].append({"use_hybrid": hybrid, "recall@1": m["recall@k"]["@1"], "recall@3": m["recall@k"]["@3"], "mrr": m["mrr"]})
    out["sweeps"] = sweeps
    out["abstention"] = abstention_table(out["baseline"]["best_similarity"], dataset, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    return out


# ------------------------------------------------------------------ live mode
class UsageTracker:
    def __init__(self, inner):
        self.inner, self.calls, self.in_chars, self.out_chars = inner, 0, 0, 0

    def generate(self, system, prompt, json_output=True, max_output_tokens=None):
        self.calls += 1
        self.in_chars += len(system) + len(prompt)
        text = self.inner.generate(system, prompt, json_output, max_output_tokens)
        self.out_chars += len(text)
        return text


def run_live(embedder, dataset: dict, settings: Settings) -> dict:
    from src.llm_service import GeminiService

    tracker = UsageTracker(GeminiService(settings))
    svc, sid, _ = build_session(settings, embedder, dataset, llm=tracker)
    rows, lat = [], []
    for q in dataset["questions"]:
        t0 = time.perf_counter()
        ans = svc.pipeline.ask(sid, q["question"])
        lat.append((time.perf_counter() - t0) * 1000)
        rows.append((q, ans))
        time.sleep(0.5)  # be gentle with rate limits
    answerable = [(q, a) for q, a in rows if q["relevant"]]
    unans = [(q, a) for q, a in rows if not q["relevant"]]
    answered = [(q, a) for q, a in answerable if a.status == AnswerStatus.ANSWERED]

    def cites_ok(q, a):
        cs = a.citations
        return sum(is_hit(type("C", (), {"filename": c.filename, "page": c.page}), q["relevant"]) for c in cs) / len(cs) if cs else 0.0

    def relevant_answer(q, a):
        low = a.answer.lower()
        return any(s.lower() in low for s in q["answer_contains_any"])

    segs = [s for _, a in answered if a.cited for s in a.cited.segments if s.support is not None]
    return {
        "n": len(rows),
        "answered_rate_on_answerable": round(len(answered) / len(answerable), 3),
        "citation_correctness": round(sum(cites_ok(q, a) for q, a in answered) / len(answered), 3) if answered else None,
        "answer_relevance_keyword_hit": round(sum(relevant_answer(q, a) for q, a in answered) / len(answered), 3) if answered else None,
        "faithfulness_proxy_mean_support": round(sum(s.support for s in segs) / len(segs), 3) if segs else None,
        "weakly_supported_segment_rate": round(sum(s.weakly_supported for s in segs) / len(segs), 3) if segs else None,
        "correct_abstention_on_unanswerable": round(sum(a.status != AnswerStatus.ANSWERED for _, a in unans) / len(unans), 3),
        "latency_ms": {"median": round(statistics.median(lat), 1), "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)},
        "usage": {"llm_calls": tracker.calls, "est_input_tokens": tracker.in_chars // 4, "est_output_tokens": tracker.out_chars // 4, "note": "chars/4 heuristic, not billing data"},
        "notes": "Keyword-hit relevance and lexical support are crude proxies; read a sample of answers manually.",
    }


def to_markdown(result: dict) -> str:
    b = result["baseline"]
    lines = [
        f"Embedder: `{result['embedder']}` (semantic: {result['is_semantic']})",
        "",
        f"Baseline ({b['n_answerable']} answerable questions): Recall@1={b['recall@k']['@1']}, Recall@3={b['recall@k']['@3']}, "
        f"Recall@5={b['recall@k']['@5']}, MRR={b['mrr']}, multi-page coverage@5={b['multi_page_coverage@5']}",
        f"Retrieval latency: median {b['retrieval_latency_ms']['median']} ms, p95 {b['retrieval_latency_ms']['p95']} ms; ingest of all documents {b['ingest_seconds']} s",
        f"Recall@3 by question type: {b['recall@3_by_type']}",
        "",
        "| chunk_size | Recall@1 | Recall@3 | MRR |", "|---|---|---|---|",
    ]
    lines += [f"| {r['chunk_size']} | {r['recall@1']} | {r['recall@3']} | {r['mrr']} |" for r in result["sweeps"]["chunk_size"]]
    lines += ["", "| hybrid | Recall@1 | Recall@3 | MRR |", "|---|---|---|---|"]
    lines += [f"| {r['use_hybrid']} | {r['recall@1']} | {r['recall@3']} | {r['mrr']} |" for r in result["sweeps"]["hybrid"]]
    lines += ["", "| threshold | false-abstain (answerable) | correct-abstain (unanswerable) |", "|---|---|---|"]
    lines += [f"| {r['threshold']} | {r['false_abstain_rate']} | {r['correct_abstain_rate']} |" for r in result["abstention"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["hashing", "sentence-transformers"], default="hashing")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--live", action="store_true", help="also run Gemini answer-quality metrics (needs GEMINI_API_KEY)")
    args = ap.parse_args()

    dataset = json.loads(DATASET.read_text())
    embedder = get_embedder(args.embedder, args.model)
    result = run_offline(embedder, embedder.name, dataset)
    if args.live:
        settings = Settings(embedding_backend=args.embedder, embedding_model=args.model)
        if not settings.has_api_key:
            sys.exit("--live requires GEMINI_API_KEY to be set.")
        result["live"] = run_live(embedder, dataset, settings)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"latest-{args.embedder}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    if "live" in result:
        print("\nLive:", json.dumps(result["live"], indent=2))
    print(f"\nFull results written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
