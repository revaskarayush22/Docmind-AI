"""Embedding backends.

* ``SentenceTransformerEmbedder`` - production backend (local model, no paid API).
* ``HashingEmbedder`` - deterministic lexical hashing vectors. It is NOT a semantic
  model; it exists so tests, offline demos and CI can run without downloading
  model weights. The UI labels it clearly when it is active.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been it this that as at by from "
    "what which who how why when where do does did can could should would will".split()
)


class Embedder(Protocol):
    name: str
    is_semantic: bool

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class EmbeddingError(RuntimeError):
    pass


class HashingEmbedder:
    """Signed feature-hashing of unigrams + bigrams, L2-normalised."""

    name = "hashing-256 (lexical, non-semantic)"
    is_semantic = False

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokens(text)
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


class SentenceTransformerEmbedder:
    """Lazy-loading wrapper around sentence-transformers (model downloads on first use)."""

    is_semantic = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Loading embedding model %s", self.model_name)
                        self._model = SentenceTransformer(self.model_name, device="cpu")
                    except Exception as exc:
                        raise EmbeddingError(
                            "The embedding model could not be loaded. Check the network connection "
                            "(the model downloads on first use) and the EMBEDDING_MODEL setting."
                        ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


_cache: dict[tuple[str, str], Embedder] = {}
_cache_lock = threading.Lock()


def get_embedder(backend: str, model_name: str) -> Embedder:
    """Return a process-wide shared embedder (so the model loads once)."""
    key = (backend, model_name)
    with _cache_lock:
        if key not in _cache:
            if backend == "hashing":
                _cache[key] = HashingEmbedder()
            elif backend == "sentence-transformers":
                _cache[key] = SentenceTransformerEmbedder(model_name)
            else:
                raise ValueError(f"Unknown embedding backend: {backend}")
        return _cache[key]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom and not math.isnan(denom) else 0.0
