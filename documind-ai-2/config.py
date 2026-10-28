"""Central configuration for DocuMind AI.

All tunables are read from environment variables (optionally loaded from a local
``.env`` file via python-dotenv). Secrets are never hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

try:  # python-dotenv is optional at import time so tests never fail on it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM -----------------------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""), repr=False)
    # Model names change often. As of Oct 2026 Google's model page listed gemini-3.8-flash as a stable
    # Flash model and recommended it (or gemini-3.5-flash-lite, cheaper) for new projects; gemini-2.5-flash
    # was restricted to existing users. Re-check https://ai.google.dev/gemini-api/docs/models before deploying.
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.8-flash"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 4096))
    llm_timeout_seconds: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 60))

    # --- Embeddings ----------------------------------------------------------
    # "sentence-transformers" (production) or "hashing" (offline demo / tests).
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- Chunking / retrieval ------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))  # characters
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    fetch_k: int = field(default_factory=lambda: _env_int("FETCH_K", 20))
    # Minimum cosine similarity of the best semantic match for the question to be
    # considered "answerable from the documents". Tune per embedding model.
    # The default depends on the embedding backend because similarity scales differ: 0.25 is a
    # starting point for MiniLM-style models (NOT yet tuned/verified here - use eval/run_eval.py
    # to tune it); the lexical hashing demo backend produces much smaller scores.
    min_similarity: float = field(
        default_factory=lambda: _env_float(
            "MIN_SIMILARITY", 0.10 if _env("EMBEDDING_BACKEND", "sentence-transformers") == "hashing" else 0.25
        )
    )
    dedupe_threshold: float = field(default_factory=lambda: _env_float("DEDUPE_THRESHOLD", 0.92))
    max_chunks_per_page: int = field(default_factory=lambda: _env_int("MAX_CHUNKS_PER_PAGE", 3))
    # If the selected documents hold at most this many characters in total (~10k tokens per 40k chars),
    # the WHOLE text is sent with every question, so summaries, "who/what is X", counts and any
    # other question work. Larger documents use search, with a broad fallback (never a refusal).
    full_context_chars: int = field(default_factory=lambda: _env_int("FULL_CONTEXT_CHARS", 40000))
    broad_context_chars: int = field(default_factory=lambda: _env_int("BROAD_CONTEXT_CHARS", 30000))
    use_hybrid: bool = field(default_factory=lambda: _env_bool("USE_HYBRID", True))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", False))
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # --- Upload limits -------------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 15))
    max_pages: int = field(default_factory=lambda: _env_int("MAX_PAGES", 200))
    max_files_per_session: int = field(default_factory=lambda: _env_int("MAX_FILES_PER_SESSION", 5))
    max_total_chunks_per_session: int = field(
        default_factory=lambda: _env_int("MAX_TOTAL_CHUNKS_PER_SESSION", 3000)
    )

    # --- Abuse protection ----------------------------------------------------
    max_question_chars: int = field(default_factory=lambda: _env_int("MAX_QUESTION_CHARS", 1000))
    questions_per_window: int = field(default_factory=lambda: _env_int("QUESTIONS_PER_WINDOW", 20))
    rate_window_seconds: int = field(default_factory=lambda: _env_int("RATE_WINDOW_SECONDS", 600))
    global_llm_calls_per_hour: int = field(default_factory=lambda: _env_int("GLOBAL_LLM_CALLS_PER_HOUR", 400))
    session_idle_minutes: int = field(default_factory=lambda: _env_int("SESSION_IDLE_MINUTES", 60))

    # --- Misc ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key.strip())

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty if OK)."""
        problems: list[str] = []
        if self.chunk_size < 200:
            problems.append("CHUNK_SIZE must be at least 200 characters.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            problems.append("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
        if self.top_k < 1 or self.fetch_k < self.top_k:
            problems.append("TOP_K must be >= 1 and FETCH_K must be >= TOP_K.")
        if not -1.0 <= self.min_similarity <= 1.0:
            problems.append("MIN_SIMILARITY must be between -1 and 1.")
        if self.embedding_backend not in {"sentence-transformers", "hashing"}:
            problems.append("EMBEDDING_BACKEND must be 'sentence-transformers' or 'hashing'.")
        return problems

    def with_overrides(self, **overrides) -> "Settings":
        """Return a copy with some values replaced (used by tests and evaluation)."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        unknown = set(overrides) - set(values)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        values.update(overrides)
        return Settings(**values)


def load_settings() -> Settings:
    return Settings()
