"""Evidence references. Citation IDs must be supplied by retrieval."""

from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from src.schemas.common import Contract, NonBlank


class DocumentSource(Contract):
    kind: Literal["document"] = "document"
    source_id: NonBlank
    document_id: NonBlank
    filename: NonBlank
    page: int = Field(ge=1, strict=True)
    chunk_id: NonBlank
    excerpt: NonBlank


class WebSource(Contract):
    kind: Literal["web"] = "web"
    source_id: NonBlank
    url: HttpUrl
    title: NonBlank
    chunk_id: NonBlank
    excerpt: NonBlank


SourceReference = Annotated[DocumentSource | WebSource, Field(discriminator="kind")]


def validate_references(
    sources: list[DocumentSource | WebSource],
    citation_groups: list[list[str]],
    source_mode: str,
) -> None:
    """Validate citation integrity; factual grounding needs a later evaluator."""
    ids = [source.source_id for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("Source IDs must be unique.")
    if any(source.kind != source_mode for source in sources):
        raise ValueError("Sources must match the selected source mode.")
    known = set(ids)
    for group in citation_groups:
        if len(set(group)) != len(group):
            raise ValueError("Repeated citations are not allowed within an item.")
        if not set(group).issubset(known):
            raise ValueError("Citation references an unknown source ID.")
