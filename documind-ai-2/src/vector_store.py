"""ChromaDB wrapper with strict per-session isolation.

Isolation model: every session gets its own Chroma collection named after its
unguessable session id, AND every query additionally filters on a ``session_id``
metadata field. Embeddings are always supplied explicitly so Chroma never
downloads its own default model.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .security import require_session_id
from .text_chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str
    score: float = 0.0  # cosine similarity for query results, 0 otherwise


def _collection_name(session_id: str) -> str:
    return f"documind_{require_session_id(session_id)}"


class VectorStore:
    def __init__(self, client=None):
        import chromadb

        # In-memory client: nothing is persisted, which matches the privacy model and
        # the ephemeral filesystem of free hosting tiers.
        self._client = client or chromadb.EphemeralClient(
            settings=chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self._lock = threading.RLock()

    # -- collections ---------------------------------------------------------
    def _get(self, session_id: str, create: bool = False):
        name = _collection_name(session_id)
        if create:
            return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        try:
            return self._client.get_collection(name=name)
        except Exception:
            return None

    # -- writes --------------------------------------------------------------
    def add_chunks(self, session_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        with self._lock:
            col = self._get(session_id, create=True)
            col.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=[e.tolist() for e in embeddings],
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "session_id": session_id,
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "page": c.page,
                        "index": c.index,
                    }
                    for c in chunks
                ],
            )

    def delete_document(self, session_id: str, doc_id: str) -> None:
        with self._lock:
            col = self._get(session_id)
            if col is not None:
                col.delete(where={"$and": [{"session_id": session_id}, {"doc_id": doc_id}]})

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            try:
                self._client.delete_collection(_collection_name(session_id))
            except Exception:
                pass  # collection did not exist

    # -- reads ---------------------------------------------------------------
    @staticmethod
    def _where(session_id: str, doc_ids: list[str] | None):
        conds: list[dict] = [{"session_id": session_id}]
        if doc_ids is not None:
            conds.append({"doc_id": {"$in": list(doc_ids)}})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def count(self, session_id: str) -> int:
        with self._lock:
            col = self._get(session_id)
            return col.count() if col is not None else 0

    def query(
        self, session_id: str, embedding: np.ndarray, n_results: int, doc_ids: list[str] | None = None
    ) -> list[StoredChunk]:
        if doc_ids is not None and not doc_ids:
            return []
        with self._lock:
            col = self._get(session_id)
            if col is None or col.count() == 0:
                return []
            n = max(1, min(n_results, col.count()))
            res = col.query(
                query_embeddings=[embedding.tolist()],
                n_results=n,
                where=self._where(session_id, doc_ids),
                include=["documents", "metadatas", "distances"],
            )
        out: list[StoredChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(self._to_chunk(cid, doc, meta, 1.0 - float(dist)))  # cosine distance -> similarity
        return out

    def all_chunks(self, session_id: str, doc_ids: list[str] | None = None) -> list[StoredChunk]:
        """Every chunk of the session (used for keyword search and study tools)."""
        if doc_ids is not None and not doc_ids:
            return []
        with self._lock:
            col = self._get(session_id)
            if col is None or col.count() == 0:
                return []
            res = col.get(where=self._where(session_id, doc_ids), include=["documents", "metadatas"])
        chunks = [self._to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.doc_id, c.index))

    @staticmethod
    def _to_chunk(cid: str, doc: str, meta: dict, score: float = 0.0) -> StoredChunk:
        return StoredChunk(
            chunk_id=cid,
            doc_id=meta["doc_id"],
            filename=meta["filename"],
            page=int(meta["page"]),
            index=int(meta["index"]),
            text=doc,
            score=score,
        )
