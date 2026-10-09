"""Quick manual test for the RAG backend."""

import sys

from rag_logic import ConfigError, PDFProcessingError, ingest_pdf, search_documents


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_backend.py <path_to_pdf>")
        return

    pdf_path = sys.argv[1]

    try:
        result = ingest_pdf(pdf_path)
        print("Ingestion result:", result)
    except ConfigError as exc:
        print("Config error:", exc)
        return
    except PDFProcessingError as exc:
        print("PDF error:", exc)
        return

    query = input("Ask a question about the PDF: ")
    for i, doc in enumerate(search_documents(query), start=1):
        page = doc.metadata.get("page", "?")
        print(f"\n--- Result {i} (page {page}) ---")
        print(doc.page_content[:300])


if __name__ == "__main__":
    main()