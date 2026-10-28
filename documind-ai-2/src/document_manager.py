"""Per-session document registry: ingestion, duplicate detection, limits, deletion, idle cleanup."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from config import Settings

from .embeddings import Embedder
from .pdf_processor import PDFProcessingError, extract_pages
from .security import (
    ValidationError,
    new_document_id,
    require_session_id,
    sanitize_filename,
    sha256_hex,
    validate_pdf_upload,
)
from .text_chunker import chunk_pages
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    sha256: str
    pages: int
    pages_with_text: int
    chunks: int
    status: str = "ready"


@dataclass
class _Session:
    docs: dict[str, DocumentRecord] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class IngestOutcome:
    filename: str
    ok: bool
    message: str
    record: DocumentRecord | None = None


class DocumentManager:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder, clock=time.monotonic):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    # -- helpers -------------------------------------------------------------
    def _session(self, session_id: str) -> _Session:
        require_session_id(session_id)
        with self._lock:
            sess = self._sessions.setdefault(session_id, _Session())
            sess.last_seen = self._clock()
            return sess

    def list_documents(self, session_id: str) -> list[DocumentRecord]:
        return list(self._session(session_id).docs.values())

    def doc_ids(self, session_id: str) -> list[str]:
        return [d.doc_id for d in self.list_documents(session_id)]

    # -- ingestion -----------------------------------------------------------
    def add_document(self, session_id: str, filename: str, data: bytes) -> IngestOutcome:
        """Validate, extract, chunk, embed and index one PDF. Never raises for user errors."""
        safe_name = sanitize_filename(filename)
        try:
            sess = self._session(session_id)
            if len(sess.docs) >= self.settings.max_files_per_session:
                raise ValidationError(
                    f"You can have at most {self.settings.max_files_per_session} documents at a time. "
                    "Remove one first."
                )
            validate_pdf_upload(data, self.settings.max_upload_bytes)
            digest = sha256_hex(data)
            for existing in sess.docs.values():
                if existing.sha256 == digest:
                    raise ValidationError(f"This file is already uploaded as “{existing.filename}”.")

            extracted = extract_pages(data, max_pages=self.settings.max_pages)
            doc_id = new_document_id()
            chunks = chunk_pages(
                extracted.pages, doc_id, safe_name, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if not chunks:
                raise ValidationError("No usable text could be extracted from this PDF.")
            if self.store.count(session_id) + len(chunks) > self.settings.max_total_chunks_per_session:
                raise ValidationError("This would exceed the amount of text allowed per session. Remove a document first.")

            embeddings = self.embedder.embed([c.text for c in chunks])
            self.store.add_chunks(session_id, chunks, embeddings)

            record = DocumentRecord(
                doc_id=doc_id,
                filename=safe_name,
                sha256=digest,
                pages=extracted.total_pages,
                pages_with_text=len(extracted.pages),
                chunks=len(chunks),
            )
            with self._lock:
                sess.docs[doc_id] = record
            note = ""
            if extracted.empty_pages:
                note = f" {extracted.empty_pages} page(s) had no extractable text (possibly scanned images) and were skipped."
            logger.info("Indexed document %s: %d pages, %d chunks", doc_id, record.pages, record.chunks)
            return IngestOutcome(safe_name, True, f"Ready: {record.pages} pages, {record.chunks} passages.{note}", record)
        except (ValidationError, PDFProcessingError) as exc:
            return IngestOutcome(safe_name, False, str(exc))
        except Exception:  # embedding / vector-store failures: log type only, never content
            logger.exception("Unexpected error while ingesting a document")
            return IngestOutcome(
                safe_name, False, "Something went wrong while processing this file. Please try again."
            )

    # -- deletion ------------------------------------------------------------
    def remove_document(self, session_id: str, doc_id: str) -> bool:
        sess = self._session(session_id)
        with self._lock:
            if doc_id not in sess.docs:
                return False
            self.store.delete_document(session_id, doc_id)
            del sess.docs[doc_id]
        return True

    def clear_session(self, session_id: str) -> None:
        require_session_id(session_id)
        with self._lock:
            self.store.delete_session(session_id)
            self._sessions.pop(session_id, None)

    def purge_idle_sessions(self) -> int:
        """Drop sessions idle longer than the configured time. Returns how many were removed."""
        cutoff = self._clock() - self.settings.session_idle_minutes * 60
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
        for sid in stale:
            self.clear_session(sid)
        return len(stale)
