"""DocuMind AI - Streamlit front end.

Design idea: the product's promise is "verify every answer", so the evidence trail is the
visual centrepiece - citation badges inside the answer text and highlighter-marked quotes
in the source list, like marking up a printed page. Everything else stays quiet.
"""
from __future__ import annotations

import logging
import re

import streamlit as st

from config import BASE_DIR, Settings, load_settings
from src.container import Services, build_services
from src.embeddings import EmbeddingError
from src.llm_service import LLMError
from src.rag_pipeline import AnswerStatus, RAGAnswer, Turn
from src.security import RateLimitExceeded, ValidationError, new_session_id, sanitize_model_markdown
from src.study_tools import StudyResult, build_study_guide_markdown

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

SAMPLE_PDF = BASE_DIR / "sample_data" / "solar_handbook.pdf"

EXAMPLE_QUESTIONS = [
    "What are the main conclusions of this document?",
    "Summarize the key requirements or recommendations.",
    "Which dates, deadlines or numbers are mentioned?",
    "What are the risks or limitations discussed?",
]

# Colour tokens: paper #F6F7F9, ink #14213D, accent blue #2B4C8C, highlighter #FFE066, muted #5B6577
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Instrument+Sans:wght@400;500;600&display=swap');

/* Set the font on the app root and form controls only - never on every element, or Streamlit's icon font breaks. */
.stApp, .stApp button, .stApp input, .stApp textarea { font-family: 'Instrument Sans', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 980px; }

/* Serif for reading material: answers, hero, study output */
h1, h2, h3, .dm-serif, [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
  font-family: 'Newsreader', Georgia, serif;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { font-size: 1.08rem; line-height: 1.65; }
[data-testid="stChatMessage"] { max-width: 46rem; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #DDE2EA; }
[data-testid="stSidebar"] h2 { font-size: 1.5rem; letter-spacing: -.01em; margin-bottom: 0; }
.dm-doc-name { font-weight: 600; font-size: .92rem; overflow-wrap: anywhere; line-height: 1.3; }
.dm-doc-meta { color: #5B6577; font-size: .8rem; }

/* Hero */
.dm-hero h1 { font-size: 3rem; line-height: 1.08; letter-spacing: -.025em; font-weight: 600; margin: 0 0 .7rem 0; color: #14213D; }
.dm-hero p { font-size: 1.15rem; line-height: 1.55; color: #3B465A; max-width: 38rem; margin: 0; }

/* The evidence trail: a sample answer with highlighter-marked quote */
.dm-sample { border: 1px solid #DDE2EA; border-radius: 10px; background: #fff; padding: 1.1rem 1.3rem; margin-top: 1.6rem; max-width: 40rem; }
.dm-sample .dm-tag { font-size: .78rem; color: #5B6577; margin-bottom: .5rem; }
.dm-sample p { font-family: 'Newsreader', Georgia, serif; font-size: 1.1rem; line-height: 1.6; margin: 0 0 .8rem 0; color: #14213D; }
.dm-chip { display: inline-block; font-family: 'Instrument Sans', sans-serif; font-size: .72rem; font-weight: 600;
  background: #E4EBF7; color: #2B4C8C; border-radius: 4px; padding: 0 .35rem; margin: 0 .1rem; vertical-align: .1em; }
.dm-quote { border-left: 4px solid #FFE066; background: #FFFBE6; padding: .5rem .8rem; font-size: .92rem; color: #3B465A; border-radius: 0 6px 6px 0; }
.dm-quote b { color: #14213D; }

/* Source quotes inside the answer */
[data-testid="stExpander"] blockquote { border-left: 4px solid #FFE066; background: #FFFBE6; padding: .45rem .8rem;
  border-radius: 0 6px 6px 0; margin: .15rem 0 .9rem 0; color: #26324A; opacity: 1 !important; }
[data-testid="stExpander"] blockquote, [data-testid="stExpander"] blockquote * { color: #26324A !important; }
[data-testid="stExpander"] blockquote p { font-size: .95rem; line-height: 1.55; margin: 0; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }

@media (max-width: 640px) { .dm-hero h1 { font-size: 2.1rem; } .block-container { padding-top: 1.2rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

MARKER_RE = re.compile(r"\[S\d+\]")

logging.basicConfig(level=load_settings().log_level)


# --------------------------------------------------------------------------- setup
def _resolve_settings() -> Settings:
    settings = load_settings()
    if not settings.has_api_key:
        try:  # hosting secrets (e.g. Streamlit Community Cloud)
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
        if key:
            settings = settings.with_overrides(gemini_api_key=str(key))
    return settings


@st.cache_resource(show_spinner="Starting DocuMind AI…")
def get_services() -> Services:
    return build_services(_resolve_settings())


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("session_id", new_session_id())
    ss.setdefault("messages", [])  # list of {"role", "content", "answer"?}
    ss.setdefault("processed_uploads", set())
    ss.setdefault("upload_nonce", 0)
    ss.setdefault("study_results", {})
    ss.setdefault("pending_question", None)


def esc(text: str) -> str:
    return sanitize_model_markdown(text)


def with_badges(markdown_text: str) -> str:
    """Turn [S1] markers into blue badges using Streamlit's own markdown (no raw HTML)."""
    return re.sub(r"\[S(\d+)\]", r" :blue-badge[S\1]", markdown_text)


def ingest(svc: Services, name: str, data: bytes) -> None:
    ss = st.session_state
    with st.spinner(f"Reading {name}…"):
        try:
            outcome = svc.manager.add_document(ss.session_id, name, data)
        except EmbeddingError as exc:
            st.error(f"**{name}**: {exc}")
            return
    if outcome.ok:
        st.toast(f"{outcome.filename}: {outcome.message}", icon="✅")
    else:
        st.error(f"**{outcome.filename}**: {outcome.message}")


# --------------------------------------------------------------------------- sidebar
def sidebar(svc: Services) -> dict:
    ss = st.session_state
    s = svc.settings
    with st.sidebar:
        st.markdown("## DocuMind AI")
        st.caption("Questions about your PDFs, answered with page citations.")

        uploads = st.file_uploader(
            f"Add PDFs (up to {s.max_upload_mb} MB and {s.max_pages} pages each)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{ss.upload_nonce}",
        )
        for up in uploads or []:
            if up.file_id in ss.processed_uploads:
                continue
            ss.processed_uploads.add(up.file_id)
            ingest(svc, up.name, up.getvalue())

        docs = svc.manager.list_documents(ss.session_id)
        st.markdown(f"##### Your documents ({len(docs)}/{s.max_files_per_session})")
        if not docs:
            st.caption("Nothing uploaded yet.")
        for d in docs:
            with st.container(border=True):
                c1, c2 = st.columns([6, 1], vertical_alignment="center")
                c1.markdown(
                    f"<div class='dm-doc-name'>{esc(d.filename)}</div>"
                    f"<div class='dm-doc-meta'>{d.pages} pages, {d.chunks} passages</div>",
                    unsafe_allow_html=True,
                )
                if c2.button("", icon=":material/close:", key=f"rm_{d.doc_id}", help="Remove this document", type="tertiary"):
                    svc.manager.remove_document(ss.session_id, d.doc_id)
                    ss.study_results = {}
                    st.rerun()

        scope = None
        if len(docs) > 1:
            names = {d.doc_id: d.filename for d in docs}
            chosen = st.multiselect("Search in", list(names), default=list(names), format_func=lambda i: names[i])
            scope = chosen or None

        detailed = st.toggle("Detailed answers", value=False, help="Longer, more structured explanations.")

        if st.button("Clear session", use_container_width=True, help="Deletes your documents and chat from this app instance."):
            svc.manager.clear_session(ss.session_id)
            ss.messages = []
            ss.study_results = {}
            ss.processed_uploads = set()
            ss.upload_nonce += 1
            ss.session_id = new_session_id()
            st.rerun()

        with st.expander("🔒 Privacy notice"):
            st.markdown(
                "- Your PDFs are processed **in memory** and indexed in a temporary, per-session store; "
                "they are not written to disk by this app.\n"
                "- To answer a question, the **most relevant text passages and your question** are sent to "
                "the Google Gemini API. Google's terms govern how it handles that data; check them "
                "before uploading sensitive documents.\n"
                "- Documents are removed when you delete them, clear the session, or after "
                f"{s.session_idle_minutes} idle minutes, or when the server restarts. "
                "This app cannot verify deletion on Google's side.\n"
                "- There are no user accounts. Each browser session is isolated, but do not upload "
                "highly confidential material to a public deployment."
            )
    return {"doc_ids": scope, "detailed": detailed, "has_docs": bool(docs)}


# --------------------------------------------------------------------------- rendering
STATUS_BADGES = {
    AnswerStatus.ANSWERED: ("Answered from your documents", "green", ":material/check_circle:"),
    AnswerStatus.INSUFFICIENT_EVIDENCE: ("Not enough evidence in your documents", "orange", ":material/search_off:"),
    AnswerStatus.OFF_TOPIC: ("Outside your documents", "gray", ":material/info:"),
    AnswerStatus.UNVERIFIED: ("Could not verify", "orange", ":material/report:"),
    AnswerStatus.NO_DOCUMENTS: ("No documents yet", "gray", ":material/info:"),
}


def render_answer(ans: RAGAnswer, key: str) -> None:
    label, color, icon = STATUS_BADGES[ans.status]
    st.badge(label, icon=icon, color=color)
    st.markdown(with_badges(esc(ans.answer)))
    if ans.limitations:
        st.warning(esc(ans.limitations), icon="⚠️")

    if ans.cited and ans.citations:
        with st.expander(f"Sources ({len(ans.citations)})", expanded=True):
            for c in ans.citations:
                st.markdown(f":blue-badge[S{c.number}] **{esc(c.filename)}**, page {c.page}")
                st.markdown(f"> {esc(c.excerpt)}")
        with st.expander("Which source supports which statement?"):
            for seg in ans.cited.segments:
                refs = ", ".join(f"S{n}" for n in seg.source_numbers) or "no citation"
                flag = " ⚠️ low word overlap" if seg.weakly_supported else ""
                plain = esc(MARKER_RE.sub("", seg.text)).strip()  # (no backslashes inside f-strings: Python 3.11)
                st.markdown(f"- {plain} → **{refs}**{flag}")
        with st.popover("📋 Copy answer"):
            st.caption("Use the copy icon at the top-right of the box.")
            st.code(ans.answer, language=None, wrap_lines=True)
    elif ans.retrieved and ans.status in (AnswerStatus.UNVERIFIED, AnswerStatus.INSUFFICIENT_EVIDENCE):
        with st.expander("Closest passages found (not used as an answer)"):
            for c in ans.retrieved[:3]:
                st.markdown(f"**{esc(c.filename)}**, page {c.page}")
                st.markdown(f"> {esc(c.text[:280])}…")
    if ans.latency_ms:
        st.caption("Took " + ", ".join(f"{k} {v / 1000:.1f}s" for k, v in ans.latency_ms.items()))


def banner(svc: Services) -> None:
    if not svc.settings.has_api_key:
        st.error(
            "No Gemini API key is set, so questions can't be answered yet. "
            "Add `GEMINI_API_KEY` to your `.env` file and restart the app. Uploading and indexing still work.",
            icon="🔑",
        )
    if not svc.embedder.is_semantic:
        st.warning(
            "Demo mode: the lexical (non-semantic) embedding backend is active, so search matches words, "
            "not meaning. Set `EMBEDDING_BACKEND=sentence-transformers` for full quality.",
            icon="🧪",
        )
    problems = svc.settings.validate()
    if problems:
        st.error("Configuration problems: " + " ".join(problems))


def landing(svc: Services) -> None:
    st.markdown(
        "<div class='dm-hero'><h1>Ask your documents.<br>Check every answer.</h1>"
        "<p>DocuMind AI reads your PDFs and answers in plain language. Each statement points to the file "
        "and page it came from. If your documents don't say, it tells you instead of guessing.</p></div>"
        "<div class='dm-sample'><div class='dm-tag'>Example of an answer</div>"
        "<p>Shade from trees or chimneys can cut a solar panel's output by up to 40 percent<span class='dm-chip'>S1</span>.</p>"
        "<div class='dm-quote'><b>S1 · solar_handbook.pdf, page 2</b><br>“Shading from trees or chimneys can reduce output by up to 40 percent.”</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.info("Upload a PDF in the sidebar to begin. Scanned or image-only PDFs need OCR first.", icon="👈")
    if SAMPLE_PDF.exists():
        st.caption("No PDF handy? Try the bundled sample.")
        if st.button("Load the sample handbook"):
            ingest(svc, SAMPLE_PDF.name, SAMPLE_PDF.read_bytes())
            st.rerun()


# --------------------------------------------------------------------------- chat
def handle_question(svc: Services, question: str, opts: dict) -> None:
    ss = st.session_state
    ss.messages.append({"role": "user", "content": question})
    history = [
        Turn(ss.messages[i]["content"], ss.messages[i + 1]["answer"].answer)
        for i in range(len(ss.messages) - 2)
        if ss.messages[i]["role"] == "user"
        and i + 1 < len(ss.messages)
        and ss.messages[i + 1]["role"] == "assistant"
        and ss.messages[i + 1].get("answer") is not None
    ]
    with st.chat_message("user"):
        st.markdown(esc(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            try:
                ans = svc.pipeline.ask(ss.session_id, question, history, opts["detailed"], opts["doc_ids"])
            except ValidationError as exc:
                st.error(str(exc))
                ss.messages.pop()
                return
            except RateLimitExceeded as exc:
                st.error(f"You're asking too quickly. {exc}")
                ss.messages.pop()
                return
            except (LLMError, EmbeddingError) as exc:
                st.error(str(exc))
                ss.messages.pop()
                return
            except Exception:
                logging.getLogger(__name__).exception("Unexpected error while answering")
                st.error("Something went wrong while answering. Please try again.")
                ss.messages.pop()
                return
        render_answer(ans, key=f"live_{len(ss.messages)}")
    ss.messages.append({"role": "assistant", "content": ans.answer, "answer": ans})


def chat_view(svc: Services, opts: dict) -> None:
    ss = st.session_state
    if not ss.messages:
        st.markdown("##### Try asking")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                ss.pending_question = q
                st.rerun()
    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(esc(m["content"]))
            else:
                render_answer(m["answer"], key=f"hist_{i}")

    question = st.chat_input("Ask a question about your documents…", max_chars=svc.settings.max_question_chars)
    question = question or ss.pending_question
    ss.pending_question = None
    if question:
        handle_question(svc, question, opts)


# --------------------------------------------------------------------------- study tools
STUDY_KINDS = {
    "summary": "Summary",
    "key_points": "Key points",
    "faq": "FAQ",
    "quiz": "Quiz",
}


def render_study(result: StudyResult) -> None:
    for w in result.warnings:
        st.warning(esc(w))
    if result.kind == "summary" and result.items:
        st.markdown(esc(result.items[0].text))
        st.caption("Sources: " + (", ".join(result.items[0].pages) or "n/a"))
    elif result.kind == "key_points":
        for it in result.items:
            st.markdown(f"- {esc(it.text)}  \n  :gray[{esc(', '.join(it.pages)) or 'no page ref'}]")
    elif result.kind == "faq":
        for it in result.items:
            with st.expander(esc(it.text)):
                st.markdown(esc(it.extra["answer"]))
                st.caption(", ".join(it.pages) or "no page ref")
    elif result.kind == "quiz":
        for n, it in enumerate(result.items, 1):
            with st.expander(f"Q{n}. {esc(it.text)}"):
                for k, o in enumerate(it.extra["options"]):
                    st.markdown(f"{chr(65 + k)}. {esc(o)}")
                with st.popover("Show answer"):
                    idx = it.extra["answer_index"]
                    st.markdown(f"**{chr(65 + idx)}. {esc(it.extra['options'][idx])}**")
                    st.markdown(esc(it.extra["explanation"]))
                    st.caption(", ".join(it.pages) or "no page ref")


def study_view(svc: Services, opts: dict) -> None:
    ss = st.session_state
    if not opts["has_docs"]:
        st.info("Upload a document to use study tools.")
        return
    st.caption("Built from passages sampled across your documents. Check anything important against the source.")
    cols = st.columns(4)
    for col, (kind, label) in zip(cols, STUDY_KINDS.items()):
        if col.button(label, key=f"gen_{kind}", use_container_width=True, help=f"Generate: {label}"):
            with st.spinner(f"Generating {label.lower()}…"):
                try:
                    ss.study_results[kind] = svc.study.generate(ss.session_id, kind, opts["doc_ids"])
                except RateLimitExceeded as exc:
                    st.error(f"You're generating too quickly. {exc}")
                except (LLMError, EmbeddingError) as exc:
                    st.error(str(exc))
                except Exception:
                    logging.getLogger(__name__).exception("Study tool failed")
                    st.error("Something went wrong generating that. Chat is unaffected; please try again.")
    for kind, label in STUDY_KINDS.items():
        if kind in ss.study_results:
            st.markdown(f"### {label}")
            render_study(ss.study_results[kind])
    if ss.study_results:
        docs = svc.manager.list_documents(ss.session_id)
        title = ", ".join(d.filename for d in docs) or "documents"
        st.download_button(
            "Download study guide (Markdown)",
            build_study_guide_markdown(title, ss.study_results),
            file_name="documind-study-guide.md",
            mime="text/markdown",
            icon=":material/download:",
        )


# --------------------------------------------------------------------------- main
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    svc = get_services()
    svc.manager.purge_idle_sessions()
    opts = sidebar(svc)
    banner(svc)
    ss = st.session_state
    if not opts["has_docs"] and not ss.messages:
        landing(svc)
        return
    if not opts["has_docs"]:
        st.info("Your documents were removed (session ended or all deleted). Upload a PDF to keep asking.")
    # A switch instead of tabs, so the chat box stays pinned to the bottom of the page.
    mode = st.segmented_control(
        "View", ["Chat", "Study tools"], default="Chat", label_visibility="collapsed", key="view_mode"
    )
    if mode == "Study tools":
        study_view(svc, opts)
    else:
        chat_view(svc, opts)


main()
