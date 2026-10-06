"""Split PDF pages without losing citation provenance."""

from collections.abc import Sequence
from hashlib import sha256

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import IngestionSettings
from src.schemas.sources import DocumentSource, WebSource


def split_pages(
    pages: Sequence[Document], settings: IngestionSettings | None = None,
) -> list[Document]:
    config = settings if settings is not None else IngestionSettings()
    if not pages:
        raise ValueError("No pages supplied for chunking.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
        add_start_index=True, length_function=len,
    )
    chunks: list[Document] = []
    seen: set[str] = set()
    for page in pages:
        for chunk in splitter.split_documents([page]):
            start = chunk.metadata["start_index"]
            if start < 0:
                raise ValueError("Cannot determine chunk position in the source page.")
            if chunk.metadata.get("kind") == "web":
                identity = f"{chunk.metadata['url']}:{start}:{chunk.page_content}"
            else:
                identity = f"{chunk.metadata['document_id']}:{chunk.metadata['page']}:{start}:{chunk.page_content}"
            chunk_id = "chunk_" + sha256(identity.encode("utf-8")).hexdigest()
            if chunk_id in seen:
                raise ValueError("Duplicate source chunks supplied.")
            seen.add(chunk_id)
            chunk.metadata.update(chunk_id=chunk_id, source_id=chunk_id)
            # Validate metadata against the citation contract before returning it.
            chunk_to_source(chunk)
            chunks.append(chunk)
            if len(chunks) > config.chunk_max_count:
                raise ValueError("Chunk count exceeds the configured limit.")
    if not chunks:
        raise ValueError("No text chunks produced.")
    return chunks


def chunk_to_source(chunk: Document) -> DocumentSource | WebSource:
    """Convert a retrieved chunk into the existing structured citation model."""
    if chunk.metadata.get("kind") == "web":
        return WebSource(source_id=chunk.metadata["source_id"], url=chunk.metadata["url"],
                         title=chunk.metadata["title"], chunk_id=chunk.metadata["chunk_id"],
                         excerpt=chunk.page_content)
    if chunk.metadata.get("kind") != "document":
        raise ValueError("Unknown evidence source kind.")
    return DocumentSource(
        source_id=chunk.metadata["source_id"],
        document_id=chunk.metadata["document_id"], filename=chunk.metadata["filename"],
        page=chunk.metadata["page"], chunk_id=chunk.metadata["chunk_id"],
        excerpt=chunk.page_content,
    )
