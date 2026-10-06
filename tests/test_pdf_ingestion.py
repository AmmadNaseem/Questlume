"""Real PDF extraction fixtures and bounded chunking checks; no network."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.core.config import IngestionSettings
from src.ingestion.pdf_loader import PDFIngestionError, PDFTextUnavailableError, load_pdfs
from src.retrieval.chunking import chunk_to_source, split_pages


def write_pdf(path, text=None, blank_first=False, encrypted=False):
    writer = PdfWriter()
    if blank_first:
        writer.add_blank_page(width=600, height=800)
    page = writer.add_blank_page(width=600, height=800)
    if text:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):
            DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 750 Td ({text}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('password')
    with path.open('wb') as output:
        writer.write(output)


class PDFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.dict(os.environ, {}, clear=True):
            self.settings = IngestionSettings(_env_file=None, pdf_upload_dir=self.root,
                                              chunk_size=50, chunk_overlap=10)

    def test_multiple_pdfs_and_chunk_provenance(self):
        first, second = self.root / 'first.pdf', self.root / 'second.pdf'
        write_pdf(first, 'Python type hints describe expected types. ' * 5, blank_first=True)
        write_pdf(second, 'Retrieval finds relevant document passages. ' * 5)
        result = load_pdfs([first, second], self.settings)
        self.assertEqual(len(result.document_ids), 2)
        self.assertEqual(result.pages[0].metadata['page'], 2)
        self.assertEqual(len(result.warnings), 1)
        chunks = split_pages(result.pages, self.settings)
        self.assertGreater(len(chunks), 2)
        self.assertEqual([c.metadata for c in chunks],
                         [c.metadata for c in split_pages(result.pages, self.settings)])
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 50)
            self.assertGreaterEqual(chunk.metadata['start_index'], 0)
            self.assertNotIn('source', chunk.metadata)
            chunk_to_source(chunk)

    def test_empty_text_and_encrypted(self):
        path = self.root / 'test.pdf'
        write_pdf(path)
        with self.assertRaises(PDFTextUnavailableError):
            load_pdfs([path], self.settings)
        write_pdf(path, 'secret', encrypted=True)
        with self.assertRaises(PDFIngestionError):
            load_pdfs([path], self.settings)

    def test_invalid_files_and_outside_root(self):
        path = self.root / 'test.pdf'
        path.write_bytes(b'not a pdf')
        for paths in ([], [path], [self.root / 'missing.pdf'], [self.root.parent / 'outside.pdf']):
            with self.assertRaises(PDFIngestionError):
                load_pdfs(paths, self.settings)

    def test_duplicate_content_and_paths(self):
        path, copy = self.root / 'one.pdf', self.root / 'two.pdf'
        write_pdf(path, 'content')
        copy.write_bytes(path.read_bytes())
        for paths in ([path, path], [path, copy]):
            with self.assertRaises(PDFIngestionError):
                load_pdfs(paths, self.settings)

    def test_limits(self):
        path = self.root / 'test.pdf'
        write_pdf(path, 'many characters here', blank_first=True)
        for update in ({'pdf_max_file_bytes': 1}, {'pdf_max_total_bytes': 1},
                       {'pdf_max_pages_per_file': 1}, {'pdf_max_extracted_chars': 1}):
            config = self.settings.model_copy(update=update)
            with self.assertRaises(PDFIngestionError):
                load_pdfs([path], config)

    def test_chunk_config_and_empty_input(self):
        with self.assertRaises(ValidationError):
            IngestionSettings(_env_file=None, chunk_size=10, chunk_overlap=10)
        with self.assertRaises(ValueError):
            split_pages([], self.settings)


if __name__ == '__main__':
    unittest.main()
