# Evaluation

## What was actually run

Everything below was produced by `python -m eval.run_eval --embedder hashing` in the development sandbox (Python 3.13). Raw output: `eval/results/latest-hashing.json` (git-ignored; regenerate it with the command).

**Important limits of these numbers**

1. **The embedder was the lexical `HashingEmbedder`, not the production sentence-transformers model.** The Hugging Face hub was not reachable from the sandbox, so the real model was never run. These results describe the chunking, hybrid retrieval and threshold logic with a word-overlap embedder; they say **nothing** about MiniLM retrieval quality.
2. **Gemini was never called** (no API key / network access in the sandbox). Citation correctness, answer faithfulness and answer relevance have therefore **not been measured**. Only retrieval and abstention-gate metrics exist.
3. **The dataset is tiny and easy**: 3 synthetic documents (9 pages, one or two chunks per page), 17 answerable and 5 unanswerable questions. Scores saturate at 1.0, so the chunk-size comparison is uninformative here. Use a larger, harder set before drawing conclusions.

## Dataset (`eval/dataset.json`)

* Documents: `solar_handbook.pdf`, `remote_work_policy.pdf`, `coral_reef_report.pdf` (3 pages each, invented facts, rendered to PDF at evaluation time).
* Questions: 15 single-page (11 lexical, 4 paraphrased), 2 multi-page, 5 unanswerable (2 unrelated, 3 topically related but absent).
* Each answerable question lists the relevant `(file, page)` pairs and strings that a correct answer should contain.

## Metrics and method

| Metric | Definition |
|---|---|
| Recall@k | Fraction of answerable questions with at least one relevant `(file, page)` in the top-k chunks (evidence gate disabled so ranking is measured in isolation) |
| MRR | Mean of 1/rank of the first relevant chunk |
| Multi-page coverage@5 | For multi-page questions, fraction of relevant pages present in the top 5 |
| False-abstain / correct-abstain | At threshold *t*, share of answerable questions whose best similarity is below *t* / share of unanswerable questions below *t* |
| Retrieval latency | Wall-clock `retrieve()` time per question (median, p95) |
| Citation correctness *(live only)* | Share of cited `(file, page)` that are in the question's relevant set |
| Answer relevance *(live only)* | Answer contains one of the expected strings (crude keyword proxy) |
| Faithfulness proxy *(live only)* | Mean word-overlap of each cited sentence with its cited passages, and share of weakly supported sentences. **Not** an entailment check |
| Latency / usage *(live only)* | End-to-end latency; call count and estimated tokens (characters / 4, not billing data) |

## Results (lexical hashing embedder, offline)

Baseline (chunk 900, overlap 150, hybrid on, 17 answerable questions): Recall@1 = 1.0, Recall@3 = 1.0, Recall@5 = 1.0, MRR = 1.0, multi-page coverage@5 = 1.0. Retrieval latency: median 6.4 ms, p95 7.0 ms (tiny corpus, 2 vCPU sandbox). Ingesting all 3 documents took 0.06 s.

| chunk_size | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| 300 | 1.0 | 1.0 | 1.0 |
| 600 | 1.0 | 1.0 | 1.0 |
| 900 | 1.0 | 1.0 | 1.0 |
| 1500 | 1.0 | 1.0 | 1.0 |

| hybrid (BM25 + semantic) | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| off | 0.882 | 1.0 | 0.941 |
| on | 1.0 | 1.0 | 1.0 |

On this dataset hybrid fusion fixed 2 of 17 first-rank misses. With 17 questions and a non-semantic embedder, treat that as an observation, not a general finding.

| similarity threshold | false-abstain (answerable) | correct-abstain (unanswerable) |
|---|---|---|
| 0.0 | 0.0 | 0.0 |
| 0.1 | 0.0 | 0.8 |
| 0.2 | 0.765 | 1.0 |
| 0.3 | 0.941 | 1.0 |

With the hashing embedder a threshold around 0.1 separates the two groups (default for that backend), but the margin is thin: the lowest best-similarity among answerable questions is 0.124, and the one unanswerable question that still passes the 0.1 gate (`u03`, the Tesla Powerwall price, which shares "battery" with the solar handbook) scores 0.104. A topically related but absent question is exactly what a similarity gate cannot reliably catch; that is what the model's `insufficient_evidence` status is for, and it is only measurable in live mode. The `0.25` default for sentence-transformers is a starting guess and **has not been tuned**.

## Reproducing and extending

```bash
# offline retrieval metrics (no key, no model download)
python -m eval.run_eval --embedder hashing

# production embedding model (needs internet for the first model download)
python -m eval.run_eval --embedder sentence-transformers

# also measure answer quality with Gemini (spends API quota, at most 22 calls)
export GEMINI_API_KEY=...        # PowerShell: $env:GEMINI_API_KEY="..."
python -m eval.run_eval --embedder sentence-transformers --live
```

After running with the real model, replace the numbers above with your own, tune `MIN_SIMILARITY` using the abstention table, and read a sample of live answers by hand; the keyword and word-overlap proxies miss paraphrase and can miss hallucinations.
