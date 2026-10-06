"""Inspect PDF ingestion locally: python -m src.ingestion.cli --pdf PATH ..."""

import argparse
import sys

from pydantic import ValidationError
from pydantic_settings import SettingsError

from src.core.config import IngestionSettings
from src.ingestion.pdf_loader import PDFIngestionError, load_pdfs
from src.retrieval.chunking import split_pages


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and chunk PDFs locally without an LLM.")
    parser.add_argument('--pdf', nargs='+', required=True, help='Paths inside data/uploads.')
    args = parser.parse_args()
    try:
        config = IngestionSettings()
        result = load_pdfs(args.pdf, config)
        chunks = split_pages(result.pages, config)
    except (ValidationError, SettingsError):
        print('Invalid ingestion configuration.', file=sys.stderr)
        return 2
    except (PDFIngestionError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f'Documents: {len(result.document_ids)}; text pages: {len(result.pages)}; chunks: {len(chunks)}')
    for warning in result.warnings:
        print('Warning:', warning)
    for document_id in result.document_ids:
        print('Document ID:', document_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
