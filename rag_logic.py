"""Core RAG logic: PDF loading, chunking, embeddings and ChromaDB storage."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Configuration ----------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_research"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 50
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


# ---------- Custom errors ----------
class ConfigError(Exception):
    """Raised when required configuration (e.g. API key) is missing."""


class PDFProcessingError(Exception):
    """Raised when a PDF is empty, corrupt, or has no extractable text."""


# ---------- Helpers ----------
def get_api_key() -> str:
    """Read the Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ConfigError("GOOGLE_API_KEY not found. Add it to your .env file.")
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

    pages = load_pdf(file_path)
    chunks = split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = name
        chunk.metadata["file_hash"] = file_hash

    # Add in batches to stay within API limits
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        vector_store.add_documents(chunks[i : i + EMBED_BATCH_SIZE])

    return {
        "file": name,
        "status": "added",
        "pages": len(pages),
        "chunks": len(chunks),
    }


# ---------- Retrieval ----------
def search_documents(query: str, k: int = 4) -> list[Document]:
    """Return the top-k most similar chunks for a query."""
    return get_vector_store().similarity_search(query, k=k)