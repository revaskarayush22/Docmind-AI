"""Security helpers: filenames, identifiers, upload validation, rate limiting, redaction."""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
import unicodedata
from collections import defaultdict, deque

PDF_MAGIC = b"%PDF-"
_SESSION_ID_RE = re.compile(r"^s[0-9a-f]{32}$")
_DOC_ID_RE = re.compile(r"^d[0-9a-f]{16}$")
_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google API key shape (classic)
    re.compile(r"AQ\.[0-9A-Za-z_\-]{20,}"),  # Google API key shape (newer "AQ." prefix)
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[=:]\s*\S+"),
]


class ValidationError(ValueError):
    """Raised when user-supplied input fails validation (safe to show to the user)."""


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: int, message: str | None = None):
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__(message or f"Rate limit reached. Try again in about {self.retry_after_seconds}s.")


def new_session_id() -> str:
    """Unguessable session identifier (128 bits of randomness)."""
    return "s" + secrets.token_hex(16)


def new_document_id() -> str:
    return "d" + secrets.token_hex(8)


def is_valid_session_id(value: str) -> bool:
    return isinstance(value, str) and bool(_SESSION_ID_RE.match(value))


def is_valid_document_id(value: str) -> bool:
    return isinstance(value, str) and bool(_DOC_ID_RE.match(value))


def require_session_id(value: str) -> str:
    if not is_valid_session_id(value):
        raise ValidationError("Invalid session identifier.")
    return value


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """Return a display-safe filename: no path components, control chars or odd symbols.

    The result is used only for display/citation - uploaded bytes are processed in
    memory and never written to a path derived from user input.
    """
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]  # strip any directory part (path traversal)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"[^\w.\- ()]", "_", name, flags=re.UNICODE)
    name = re.sub(r"\.{2,}", ".", name).strip(" .")
    if not name:
        name = "document.pdf"
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, "pdf"
    if ext.lower() != "pdf":
        stem, ext = name, "pdf"
    stem = stem[: max(1, max_length - 4)]
    return f"{stem}.pdf"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_upload(data: bytes, max_bytes: int) -> None:
    """Cheap structural checks before parsing. Raises ValidationError."""
    if not data:
        raise ValidationError("The file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    # Real PDFs may have a few bytes of junk before the header; accept within the first 1 KiB.
    if PDF_MAGIC not in data[:1024]:
        raise ValidationError("This does not look like a PDF file (missing PDF header).")


def redact(text: str) -> str:
    """Remove obvious secrets from a string before it is logged or shown."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


_IMAGE_MD = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_MD = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_RAW_URL = re.compile(r"https?://\S+")
_HTML_TAG = re.compile(r"</?[A-Za-z!][^>]{0,200}>")  # real tags only; keeps "x < 5 and y > 3"


def sanitize_model_markdown(text: str) -> str:
    """Neutralise markup in model/document-derived text before rendering.

    A malicious PDF could make the model emit a markdown image or link whose URL
    carries data out to a third party when rendered. We drop images, flatten links to
    their label, defuse raw URLs, strip HTML tags and escape ``$`` (LaTeX in Streamlit).
    """
    text = _IMAGE_MD.sub(r"\1", text)
    text = _LINK_MD.sub(r"\1", text)
    text = _RAW_URL.sub(lambda m: m.group(0).replace("http", "hxxp", 1), text)
    text = _HTML_TAG.sub("", text)
    return text.replace("$", r"\$")


class SlidingWindowLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by an arbitrary string."""

    def __init__(self, max_events: int, window_seconds: int, clock=time.monotonic):
        self.max_events = max_events
        self.window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record one event for ``key`` or raise RateLimitExceeded."""
        now = self._clock()
        with self._lock:
            q = self._events[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.max_events:
                raise RateLimitExceeded(self.window - (now - q[0]))
            q.append(now)

    def forget(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)
