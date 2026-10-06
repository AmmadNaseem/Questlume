"""Local PDF search demo; no generation provider calls."""

import argparse
import sys

from src.core.config import IngestionSettings
from src.ingestion.pdf_loader import load_pdfs
from src.retrieval.chunking import chunk_to_source, split_pages
from src.retrieval.embeddings import get_embeddings
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import build_document_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Search selected PDFs with local embeddings.")
    parser.add_argument("--pdf", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    try:
        settings = IngestionSettings()
        result = load_pdfs(args.pdf, settings)
        chunks = split_pages(result.pages, settings)
        index = build_document_index(chunks, result.document_ids, get_embeddings(settings))
        documents = build_retriever(index, settings).invoke(args.query)
    except Exception as error:
        # Model download/library errors can contain paths or tokens; avoid raw dumps.
        print(f"Local retrieval failed ({type(error).__name__}). Check dependencies, PDFs, and model cache.", file=sys.stderr)
        return 2
    for warning in result.warnings:
        print("Warning:", warning)
    for document in documents:
        source = chunk_to_source(document)
        print(f"\n{source.filename}, page {source.page} [{source.source_id}]\n{source.excerpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
