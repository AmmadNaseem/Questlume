"""Bounded local PDF ingestion. Never calls an LLM or uploads document text."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from src.core.config import IngestionSettings


class PDFIngestionError(ValueError):
    """A file cannot be safely accepted or extracted."""


class PDFTextUnavailableError(PDFIngestionError):
    """No text could be extracted; OCR or another extractor may be needed."""


@dataclass(frozen=True)
class PDFLoadResult:
    pages: tuple[Document, ...]
    document_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def load_pdfs(
    paths: Sequence[str | Path], settings: IngestionSettings | None = None,
) -> PDFLoadResult:
    """Load files confined to the upload directory, with one-based page metadata.

    All-or-nothing batch semantics: invalid files raise rather than silently
    disappearing. Empty pages are skipped with warnings; wholly empty files fail.
    Paths are trusted server-side references, not public request parameters.
    """
    config = settings if settings is not None else IngestionSettings()
    if isinstance(paths, (str, bytes, Path)) or not paths:
        raise PDFIngestionError("Supply a nonempty sequence of PDF paths.")
    if len(paths) > config.pdf_max_files:
        raise PDFIngestionError("PDF file count exceeds the configured limit.")
    root = config.pdf_upload_dir.resolve()
    files: list[Path] = []
    total_bytes = 0
    for item in paths:
        path = Path(item).resolve()
        if not path.is_relative_to(root):
            raise PDFIngestionError("PDF files must be inside the upload directory.")
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise PDFIngestionError("Each input must be an existing PDF file.")
        size = path.stat().st_size
        if not size or size > config.pdf_max_file_bytes:
            raise PDFIngestionError("PDF size exceeds limits or the file is empty.")
        total_bytes += size
        if total_bytes > config.pdf_max_total_bytes:
            raise PDFIngestionError("Total PDF upload size exceeds the configured limit.")
        if path in files:
            raise PDFIngestionError("Duplicate PDF paths are not allowed.")
        files.append(path)

    pages: list[Document] = []
    ids: list[str] = []
    warnings: list[str] = []
    extracted_chars = 0
    for path in files:
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise PDFIngestionError("File is not a recognizable PDF.")
            stream.seek(0)
            digest = sha256()
            for block in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(block)
        document_id = "doc_" + digest.hexdigest()
        if document_id in ids:
            raise PDFIngestionError("Duplicate PDF content is not allowed.")
        start = len(pages)
        try:
            with path.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted:
                    raise PDFIngestionError("Encrypted PDFs are not supported.")
                if len(reader.pages) > config.pdf_max_pages_per_file:
                    raise PDFIngestionError("PDF page count exceeds the configured limit.")
            for page in PyPDFLoader(str(path), mode="page").lazy_load():
                number = page.metadata["page"] + 1
                text = page.page_content.replace("\x00", "").strip()
                if not text:
                    warnings.append(f"{path.name}: page {number} has no extractable text; skipped.")
                    continue
                extracted_chars += len(text)
                if extracted_chars > config.pdf_max_extracted_chars:
                    raise PDFIngestionError("Extracted text exceeds the configured batch limit.")
                pages.append(Document(page_content=text, metadata={
                    "kind": "document", "document_id": document_id,
                    "filename": path.name, "page": number,
                    "page_label": str(page.metadata.get("page_label", number)),
                }))
        except PDFIngestionError:
            raise
        except Exception:
            raise PDFIngestionError(f"Cannot parse PDF: {path.name}.") from None
        if len(pages) == start:
            raise PDFTextUnavailableError(
                f"No extractable text in {path.name}; OCR or another extraction method is required."
            )
        ids.append(document_id)
    return PDFLoadResult(tuple(pages), tuple(ids), tuple(warnings))
