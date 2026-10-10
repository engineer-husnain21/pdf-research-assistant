"""Core RAG logic: PDF ingestion, ChromaDB storage and conversational QA."""

import contextvars
import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Configuration ----------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_research"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 20
BATCH_PAUSE_SECONDS = 12
MAX_EMBED_RETRIES = 4
RETRY_WAIT_SECONDS = 40
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-3.8-flash"
CONDENSE_MODEL = "gemini-3.1-flash-lite"
FALLBACK_CHAT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]
TOP_K = 4
MAX_HISTORY_MESSAGES = 8
FALLBACK_ANSWER = (
    "I couldn't find this in the uploaded documents. "
    "Try rephrasing your question or upload a PDF that covers this topic."
)
NO_DOCUMENTS_ANSWER = "Please upload and process a PDF first, then ask your question."


# ---------- Custom errors ----------
class ConfigError(Exception):
    """Raised when required configuration (e.g. API key) is missing."""


class PDFProcessingError(Exception):
    """Raised when a PDF is empty, corrupt, or has no extractable text."""


class LLMUnavailableError(Exception):
    """Raised when every configured chat model is out of quota or unavailable."""


# ---------- Helpers ----------
# Per-session key supplied through the UI. A ContextVar keeps each Streamlit
# user session isolated, so one visitor's key is never used for another.
_runtime_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "runtime_api_key", default=None
)


def set_runtime_api_key(key: str | None) -> None:
    """Set (or clear) an API key that overrides the one from .env."""
    _runtime_api_key.set(key.strip() if key else None)


def get_api_key() -> str:
    """Return the UI-provided key if present, otherwise the one from .env."""
    api_key = _runtime_api_key.get() or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ConfigError(
            "No API key found. Add GOOGLE_API_KEY to your .env file "
            "or paste a key in the sidebar."
        )
    return api_key

def compute_file_hash(file_path: str) -> str:
    """Return the SHA-256 hash of a file (used to avoid duplicate ingestion)."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ---------- Ingestion pipeline ----------
def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF and return one Document per page.

    Raises PDFProcessingError for missing, corrupt, or text-less PDFs.
    """
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        raise PDFProcessingError(f"'{path.name}' is missing or empty.")

    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise PDFProcessingError(
            f"'{path.name}' could not be read (corrupt or encrypted PDF): {exc}"
        ) from exc

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise PDFProcessingError(
            f"No extractable text in '{path.name}'. It may be a scanned PDF "
            "(image-only) which needs OCR."
        )
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    """Split page documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embeddings client."""
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        google_api_key=get_api_key(),
    )


def get_vector_store() -> Chroma:
    """Open (or create) the persistent ChromaDB collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def is_already_ingested(vector_store: Chroma, file_hash: str) -> bool:
    """Check whether a PDF with this hash is already in the database."""
    existing = vector_store.get(where={"file_hash": file_hash}, limit=1)
    return len(existing["ids"]) > 0


def ingest_pdf(file_path: str, display_name: str | None = None) -> dict:
    """Full pipeline: load -> split -> embed -> store in ChromaDB.

    Returns a small summary dict that the UI can show to the user.
    """
    name = display_name or Path(file_path).name
    vector_store = get_vector_store()
    file_hash = compute_file_hash(file_path)

    if is_already_ingested(vector_store, file_hash):
        return {"file": name, "status": "skipped", "chunks": 0}

    try:
        pages = load_pdf(file_path)
    except PDFProcessingError as exc:
        # Show the user's real file name instead of the temporary one
        raise PDFProcessingError(str(exc).replace(Path(file_path).name, name)) from exc
    chunks = split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = name
        chunk.metadata["file_hash"] = file_hash

    # Add in small batches with pauses; clean up if the upload fails midway
    try:
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            for attempt in range(MAX_EMBED_RETRIES + 1):
                try:
                    vector_store.add_documents(batch)
                    break
                except Exception as exc:
                    quota_hit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                    if not quota_hit or attempt == MAX_EMBED_RETRIES:
                        raise
                    time.sleep(RETRY_WAIT_SECONDS)
            if start + EMBED_BATCH_SIZE < len(chunks):
                time.sleep(BATCH_PAUSE_SECONDS)
    except Exception:
        # Remove the partial upload so the file can be processed again cleanly
        vector_store.delete(where={"file_hash": file_hash})
        raise

    return {
        "file": name,
        "status": "added",
        "pages": len(pages),
        "chunks": len(chunks),
    }


CANDIDATE_POOL = 30
RERANK_STOPWORDS = {
    "what", "who", "when", "where", "which", "that", "this", "with", "from",
    "have", "been", "were", "about", "does", "give", "gave", "tell", "into",
    "their", "there", "these", "those", "would", "could", "should",
}


def search_documents(query: str, k: int = 4) -> list[Document]:
    """Return the top-k chunks, re-ranked by how many query keywords they contain."""
    candidates = get_vector_store().similarity_search(query, k=max(k, CANDIDATE_POOL))

    terms = [w.strip("?.,!:;'\"()").lower() for w in query.split()]
    terms = [w for w in terms if len(w) > 3 and w not in RERANK_STOPWORDS]

    def sort_key(item):
        position, doc = item
        text = doc.page_content.lower()
        matches = sum(1 for term in terms if term in text)
        return (-matches, position)  # more keyword matches first, then original order

    ranked = sorted(enumerate(candidates), key=sort_key)
    return [doc for _, doc in ranked[:k]]

def has_documents() -> bool:
    """Return True if at least one chunk exists in the vector database."""
    return len(get_vector_store().get(limit=1)["ids"]) > 0


def list_ingested_documents() -> dict[str, int]:
    """Return {file name: chunk count} for everything stored in ChromaDB."""
    data = get_vector_store().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"] or []:
        name = (meta or {}).get("source", "Unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


# ---------- Prompts ----------
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question so it can be understood without the history. Do NOT answer "
            "it. If it is already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful research assistant. Answer the question using ONLY "
            "the context below, which comes from the user's PDF documents. "
            "Start directly with the answer. Never begin with phrases like "
            "'Based on the provided documents', 'According to the context' or "
            "'The documents state'. Do not mention the context or documents at all. "
            "Be clear and concise, and use short bullet points when helpful. "
            "If the context does not contain the answer, reply with exactly this "
            f"sentence and nothing else: {FALLBACK_ANSWER}\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


# ---------- LLM with automatic model fallback ----------
def get_model_candidates() -> list[str]:
    """Return chat models in priority order: .env choice first, then fallbacks."""
    preferred = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return list(dict.fromkeys([preferred, *FALLBACK_CHAT_MODELS]))


def get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat client. Fails fast so fallbacks kick in quickly."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=get_api_key(),
        temperature=0.2,
        max_retries=1,
        timeout=30,
    )


def _is_recoverable_llm_error(exc: Exception) -> bool:
    """True for quota, overload and 'model not found' errors (try next model)."""
    text = str(exc)
    markers = (
        "RESOURCE_EXHAUSTED",
        "429",
        "NOT_FOUND",
        "404",
        "UNAVAILABLE",
        "503",
        "overloaded",
        "no longer available",
    )
    return any(marker in text for marker in markers)


def _invoke_with_fallback(prompt, inputs: dict, models: list[str]) -> str:
    """Run a prompt on each model in turn until one succeeds."""
    last_error: Exception | None = None
    for model_name in models:
        try:
            chain = prompt | get_llm(model_name) | StrOutputParser()
            return chain.invoke(inputs).strip()
        except ConfigError:
            raise
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                last_error = exc
                continue
            raise
    raise LLMUnavailableError(
        "All chat models are currently out of quota or unavailable."
    ) from last_error


# ---------- Conversational Retrieval ----------
def to_langchain_history(messages: list[dict]) -> list[BaseMessage]:
    """Convert UI messages ({'role', 'content'}) to LangChain messages.

    Only the most recent messages are kept to control token usage.
    """
    history: list[BaseMessage] = []
    for msg in messages[-MAX_HISTORY_MESSAGES:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


SNIPPET_STOPWORDS = {
    "what", "who", "when", "where", "which", "that", "this", "with", "from",
    "have", "been", "were", "about", "does", "give", "gave", "tell", "into",
}


def _best_snippet(text: str, query: str | None, size: int = 380) -> str:
    """Return a text window around the first query keyword found in the chunk."""
    flat = " ".join(text.split())
    start = 0
    if query:
        words = [w.strip("?.,!'\"").lower() for w in query.split()]
        words = [w for w in words if len(w) > 3 and w not in SNIPPET_STOPWORDS]
        lowered = flat.lower()
        hits = [lowered.find(w) for w in words if lowered.find(w) != -1]
        if hits:
            start = max(0, min(hits) - 100)
    prefix = "…" if start > 0 else ""
    return prefix + flat[start : start + size]


def format_sources(docs: list[Document], query: str | None = None) -> list[dict]:
    """Build a de-duplicated list of source info for the UI."""
    sources, seen = [], set()
    for doc in docs:
        name = doc.metadata.get("source", "Unknown")
        page = int(doc.metadata.get("page", 0)) + 1  # PyPDF pages start at 0
        if (name, page) in seen:
            continue
        seen.add((name, page))
        sources.append(
            {
                "source": name,
                "page": page,
                "snippet": _best_snippet(doc.page_content, query),
            }
        )
    return sources


def build_passages_answer(docs: list[Document]) -> str:
    """Fallback answer used when no LLM is available (no API call needed)."""
    sources = format_sources(docs)[:3]
    pages = ", ".join(f"page {s['page']}" for s in sources)
    return (
        "I can't generate a written answer right now, but I found the most "
        f"relevant passages in your documents ({pages}). "
        "You can read them in the source cards below."
    )


def answer_question(question: str, chat_history: list[dict] | None = None) -> dict:
    """Answer a question using the PDFs, with conversation memory.

    Returns: {"answer": str, "sources": list[dict], "found": bool, "degraded": bool}.
    "degraded" is True when the LLM was unavailable and raw passages were returned.
    """
    if not has_documents():
        return {
            "answer": NO_DOCUMENTS_ANSWER,
            "sources": [],
            "found": False,
            "degraded": False,
        }

    history = to_langchain_history(chat_history or [])

    # Step 1: turn a follow-up question into a standalone question
    standalone = question
    if history:
        condense_models = list(dict.fromkeys([CONDENSE_MODEL, *get_model_candidates()]))
        try:
            standalone = (
                _invoke_with_fallback(
                    CONDENSE_PROMPT,
                    {"chat_history": history, "question": question},
                    condense_models,
                )
                or question
            )
        except LLMUnavailableError:
            standalone = question  # still try retrieval with the raw question

    # Step 2: retrieve relevant chunks
    docs = search_documents(standalone, k=TOP_K)
    if not docs:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "found": False,
            "degraded": False,
        }

    context = "\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')}, page "
        f"{int(d.metadata.get('page', 0)) + 1}]\n{d.page_content}"
        for d in docs
    )

    # Step 3: generate the grounded answer (or degrade to raw passages)
    try:
        answer = _invoke_with_fallback(
            ANSWER_PROMPT,
            {"context": context, "chat_history": history, "question": question},
            get_model_candidates(),
        )
    except LLMUnavailableError:
        return {
            "answer": build_passages_answer(docs),
            "sources": format_sources(docs, standalone) if found else [],
            "found": True,
            "degraded": True,
        }

    found = FALLBACK_ANSWER not in answer
    return {
        "answer": answer,
        "sources": format_sources(docs, standalone) if found else [],
        "found": found,
        "degraded": False,
    }