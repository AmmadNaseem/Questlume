"""Build an in-memory FAISS index from an explicit PDF selection."""

from collections.abc import Sequence
from copy import deepcopy

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.chunking import chunk_to_source


def build_document_index(
    chunks: Sequence[Document],
    document_ids: Sequence[str],
    embeddings: Embeddings,
) -> FAISS:
    """Exclude unselected documents BEFORE embedding and indexing.

    Document selection is not authorization: the future API must establish
    ownership before passing IDs here. No pickle loading or disk persistence.
    """
    if isinstance(document_ids, str) or not document_ids:
        raise ValueError("Select at least one document ID.")
    selected = set(document_ids)
    if any(not isinstance(value, str) or not value.strip() for value in document_ids):
        raise ValueError("Document IDs must be nonblank strings.")
    if len(selected) != len(document_ids):
        raise ValueError("Document IDs must be unique.")
    documents: list[Document] = []
    chunk_ids: set[str] = set()
    found: set[str] = set()
    for chunk in chunks:
        if chunk.metadata.get("document_id") not in selected:
            continue
        if chunk.metadata.get("kind") != "document":
            raise ValueError("PDF indexes only accept document chunks.")
        source = chunk_to_source(chunk)
        if source.chunk_id in chunk_ids:
            raise ValueError("Duplicate chunk IDs are not allowed.")
        chunk_ids.add(source.chunk_id)
        found.add(source.document_id)
        documents.append(deepcopy(chunk))
    if found != selected:
        raise ValueError("One or more selected documents have no indexed chunks.")
    return FAISS.from_documents(
        documents, embeddings, ids=[doc.metadata["chunk_id"] for doc in documents]
    )


def build_web_index(chunks: Sequence[Document], embeddings: Embeddings) -> FAISS:
    """Index only fetched web chunks; never mix in PDF content."""
    from src.schemas.sources import WebSource
    if not chunks:
        raise ValueError('No web chunks supplied.')
    ids: set[str] = set()
    for chunk in chunks:
        source = chunk_to_source(chunk)
        if not isinstance(source, WebSource):
            raise ValueError('Web index only accepts web evidence.')
        if source.chunk_id in ids:
            raise ValueError('Duplicate web chunk IDs.')
        ids.add(source.chunk_id)
    return FAISS.from_documents([deepcopy(chunk) for chunk in chunks], embeddings,
                                ids=[chunk.metadata['chunk_id'] for chunk in chunks])
