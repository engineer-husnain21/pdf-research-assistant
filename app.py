"""Streamlit UI for the Personalized PDF Research Assistant."""

import html
import os
import tempfile

import streamlit as st

from rag_logic import (
    ConfigError,
    PDFProcessingError,
    answer_question,
    get_api_key,
    ingest_pdf,
    list_ingested_documents,
    set_runtime_api_key,
)

st.set_page_config(
    page_title="PDF Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
CUSTOM_CSS = """
<style>
.stApp {
    background:
        radial-gradient(1100px 600px at 8% -10%, rgba(139, 92, 246, 0.20), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(34, 211, 238, 0.12), transparent 55%),
        #0B0F1A;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 960px; }

section[data-testid="stSidebar"] {
    background: rgba(18, 24, 41, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.hero { padding: 8px 0 22px 0; }
.hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: 0;
    background: linear-gradient(90deg, #A78BFA 0%, #22D3EE 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { color: #9CA3AF; font-size: 1.02rem; margin-top: 6px; }
.chip-row { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    font-size: 0.78rem; padding: 4px 12px; border-radius: 999px;
    color: #C4B5FD; background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
}
.status-pill .dot { width: 9px; height: 9px; border-radius: 50%; }
.status-ready { background: rgba(16, 185, 129, 0.12); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.35); }
.status-ready .dot { background: #10B981; box-shadow: 0 0 8px #10B981; }
.status-empty { background: rgba(245, 158, 11, 0.12); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.35); }
.status-empty .dot { background: #F59E0B; }
.status-error { background: rgba(239, 68, 68, 0.12); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.35); }
.status-error .dot { background: #EF4444; }

.kb-title { color: #9CA3AF; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; margin: 18px 0 8px 0; }
.kb-item {
    display: flex; justify-content: space-between; align-items: center; gap: 8px;
    padding: 9px 12px; margin-bottom: 6px; border-radius: 10px;
    background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.07);
    font-size: 0.85rem; color: #E5E7EB;
}
.kb-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kb-count { color: #A78BFA; font-size: 0.75rem; white-space: nowrap; }

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px; padding: 14px 18px; margin-bottom: 12px;
}

.src-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
.src-card {
    padding: 12px 14px; border-radius: 12px;
    background: rgba(139, 92, 246, 0.07); border: 1px solid rgba(139, 92, 246, 0.25);
}
.src-head { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.src-file { font-weight: 600; font-size: 0.82rem; color: #DDD6FE; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.src-page {
    font-size: 0.72rem; padding: 2px 9px; border-radius: 999px; white-space: nowrap;
    color: #67E8F9; background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3);
}
.src-snippet { font-size: 0.8rem; color: #9CA3AF; line-height: 1.5; }

.stButton > button {
    border-radius: 12px; color: #FFFFFF; transition: all 0.2s ease;
    border: 1px solid rgba(139, 92, 246, 0.45);
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.28), rgba(34, 211, 238, 0.12));
}
.stButton > button:hover {
    transform: translateY(-1px); border-color: #8B5CF6;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.28);
}
.limited-badge {
    display: inline-block; margin-bottom: 10px; padding: 4px 12px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; color: #FCD34D;
    background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_PROMPTS = [
    "Summarize the main points of the document",
    "What are the key topics covered?",
    "List important names, dates or figures mentioned",
]


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = None

# Apply the visitor's own key (if any) on every rerun, before anything uses the API.
set_runtime_api_key(st.session_state.user_api_key)


def set_pending_question(text: str) -> None:
    """Callback for example-prompt buttons."""
    st.session_state.pending_question = text


def clear_chat() -> None:
    """Callback for the clear-chat button."""
    st.session_state.messages = []

def apply_user_key() -> None:
    """Callback: store the key typed in the sidebar for this session only."""
    st.session_state.user_api_key = st.session_state.api_key_input.strip() or None
    st.session_state.api_key_input = ""


def remove_user_key() -> None:
    """Callback: forget the key entered in the sidebar."""
    st.session_state.user_api_key = None


# ---------- UI helpers ----------
def render_status_pill(kind: str, text: str) -> None:
    st.markdown(
        f'<span class="status-pill status-{kind}"><span class="dot"></span>{text}</span>',
        unsafe_allow_html=True,
    )


def friendly_error(exc: Exception) -> str:
    """Turn raw API errors into short, human-readable messages."""
    text = str(exc)
    lowered = text.lower()
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "The AI service has reached its free usage limit. Please try again in a little while."
    if "API_KEY_INVALID" in text or "PERMISSION_DENIED" in text or "API key" in text:
        return "The API key looks invalid. Please check GOOGLE_API_KEY in your .env file."
    if "timed out" in lowered or "timeout" in lowered or "connect" in lowered:
        return "Could not reach the AI service. Please check your internet connection and try again."
    return "Something unexpected went wrong. Please try again."


def render_limited_badge() -> None:
    st.markdown(
        '<span class="limited-badge">⚡ Limited mode: AI quota reached, '
        "showing passages from your documents</span>",
        unsafe_allow_html=True,
    )

def render_sources(sources: list[dict]) -> None:
    """Show source cards (file, page number, snippet) inside an expander."""
    if not sources:
        return
    cards = "".join(
        '<div class="src-card">'
        '<div class="src-head">'
        f'<span class="src-file">📄 {html.escape(s["source"])}</span>'
        f'<span class="src-page">Page {s["page"]}</span>'
        "</div>"
        f'<div class="src-snippet">{html.escape(s["snippet"])}…</div>'
        "</div>"
        for s in sources
    )
    with st.expander(f"📚 Sources ({len(sources)})"):
        st.markdown(f'<div class="src-grid">{cards}</div>', unsafe_allow_html=True)


def process_upload(uploaded_file) -> dict:
    """Save an uploaded PDF to a temp file and run the ingestion pipeline."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        return ingest_pdf(tmp_path, display_name=uploaded_file.name)
    finally:
        os.remove(tmp_path)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📚 Research Assistant")
    st.caption("Upload PDFs, process them, then chat with your documents.")

    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⚡ Process documents", disabled=not files, width="stretch"):
        added, skipped, failed = 0, 0, 0
        with st.status("Processing documents...", expanded=True) as status:
            for uploaded in files:
                st.write(f"Reading **{uploaded.name}** ...")
                try:
                    result = process_upload(uploaded)
                except ConfigError as exc:
                    st.write(f"❌ {exc}")
                    failed += 1
                except PDFProcessingError as exc:
                    st.write(f"❌ {exc}")
                    failed += 1
                except Exception as exc:  # API / network problems
                    st.write(f"❌ Could not process '{uploaded.name}': {friendly_error(exc)}")
                    failed += 1
                else:
                    if result["status"] == "skipped":
                        st.write("↪️ Already in the knowledge base, skipped.")
                        skipped += 1
                    else:
                        st.write(
                            f"✅ {result['pages']} pages → {result['chunks']} chunks stored."
                        )
                        added += 1
            final_state = "error" if failed and not (added or skipped) else "complete"
            status.update(
                label=f"Done: {added} added, {skipped} skipped, {failed} failed",
                state=final_state,
                expanded=False,
            )

    # Status indicator + knowledge base list
    try:
        get_api_key()
        documents = list_ingested_documents()
        api_ok = True
    except ConfigError:
        documents, api_ok = {}, False
    except Exception:
        documents, api_ok = {}, True

    st.markdown("---")
    if not api_ok:
        render_status_pill("error", "API key missing: add one below")
    elif documents:
        render_status_pill("ready", "Ready to chat")
    else:
        render_status_pill("empty", "No documents yet")

    if documents:
        st.markdown('<div class="kb-title">Knowledge base</div>', unsafe_allow_html=True)
        items = "".join(
            '<div class="kb-item">'
            f'<span class="kb-name">📄 {html.escape(name)}</span>'
            f'<span class="kb-count">{count} chunks</span>'
            "</div>"
            for name, count in documents.items()
        )
        st.markdown(items, unsafe_allow_html=True)

    st.markdown("---")
    st.button(
        "🧹 Clear chat",
        on_click=clear_chat,
        disabled=not st.session_state.messages,
        width="stretch",
    )

    with st.expander("🔑 Use your own API key"):
        st.caption(
            "Get a free key from Google AI Studio. It is kept only in this "
            "browser session and is never saved to disk."
        )
        if st.session_state.user_api_key:
            st.success("Using your key for this session.")
            st.button("Remove key", on_click=remove_user_key, width="stretch")
        else:
            st.text_input(
                "Gemini API key",
                type="password",
                key="api_key_input",
                label_visibility="collapsed",
                placeholder="Paste your Gemini API key",
            )
            st.button("Use this key", on_click=apply_user_key, width="stretch")


# ---------- Main area: header ----------
st.markdown(
    '<div class="hero">'
    '<h1 class="hero-title">PDF Research Assistant</h1>'
    '<div class="hero-sub">Ask questions about your documents and get answers with page-level sources.</div>'
    '<div class="chip-row">'
    '<span class="chip">RAG</span><span class="chip">Gemini</span>'
    '<span class="chip">ChromaDB</span><span class="chip">LangChain</span>'
    "</div></div>",
    unsafe_allow_html=True,
)

# ---------- Main area: chat ----------
prompt = st.chat_input("Ask anything about your PDFs...")
if not prompt:
    prompt = st.session_state.pop("pending_question", None)

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("degraded"):
            render_limited_badge()
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

if not st.session_state.messages and not prompt:
    if documents:
        st.markdown("##### Try asking")
        columns = st.columns(len(EXAMPLE_PROMPTS))
        for column, example in zip(columns, EXAMPLE_PROMPTS):
            column.button(
                example,
                key=f"example_{example}",
                on_click=set_pending_question,
                args=(example,),
                width="stretch",
            )
    else:
        st.info("👈 Upload a PDF in the sidebar and click **Process documents** to get started.")

if prompt:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        try:
            with st.spinner("Searching your documents..."):
                result = answer_question(prompt, history)
        except ConfigError as exc:
            st.session_state.messages.pop()
            st.error(f"Configuration error: {exc}")
        except Exception as exc:
            st.session_state.messages.pop()
            st.error(friendly_error(exc))
        else:
            if result.get("degraded"):
                render_limited_badge()
            st.markdown(result["answer"])
            render_sources(result["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "degraded": result.get("degraded", False),
                }
            )