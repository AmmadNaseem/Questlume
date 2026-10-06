"""Request isolation and source/output routing without external API calls."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from src.core.config import Settings
from src.ingestion.pdf_loader import PDFIngestionError, PDFLoadResult
from src.ingestion.web_loader import WebLoadResult
from src.schemas.requests import DocumentInput, InterviewRequest, PresentationRequest, WebInput
from src.services.generation import Upload, generate, staged_pdfs
from src.pipelines.interview import InsufficientEvidenceError


class GenerationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.dict(os.environ, {}, clear=True):
            self.settings = Settings(_env_file=None, llm_provider='groq', llm_fallback_providers=(),
                groq_model='fake', groq_api_key='fake', pdf_upload_dir=Path(self.temp.name))
        self.upload = Upload('notes.pdf', b'%PDF-fake')

    def request(self, source):
        return InterviewRequest(source=source, topic='Types', job_role='Engineer',
            experience_level='senior', difficulty='advanced', question_count=2, question_types=['conceptual'])

    def test_staging_is_isolated_and_cleans_up_on_failure(self):
        with self.assertRaises(RuntimeError):
            with staged_pdfs([self.upload], self.settings) as paths:
                path = paths[0]
                self.assertEqual(path.name, 'notes.pdf')
                self.assertTrue(path.is_relative_to(self.settings.pdf_upload_dir.resolve()))
                with staged_pdfs([self.upload], self.settings) as other:
                    self.assertNotEqual(path.parent, other[0].parent)
                raise RuntimeError('Simulated downstream failure')
        self.assertFalse(path.exists())
        self.assertEqual(list(self.settings.pdf_upload_dir.iterdir()), [])

    def test_upload_boundaries(self):
        for uploads in ([], [Upload('../bad.pdf', b'%PDF-')], [Upload('a.pdf', b'not pdf')],
                        [self.upload, self.upload], [Upload('a.pdf', b'')], [Upload('CON.pdf', b'%PDF-')]):
            with self.assertRaises(PDFIngestionError):
                with staged_pdfs(uploads, self.settings):
                    self.fail('Invalid upload was accepted')
        small = self.settings.model_copy(update={'pdf_max_total_bytes': 3})
        with self.assertRaises(PDFIngestionError):
            with staged_pdfs([self.upload], small):
                self.fail('Oversized batch accepted')

    def test_invalid_source_or_question_count_never_starts_models(self):
        with patch('src.services.generation.get_llm') as llm:
            with self.assertRaises(ValueError):
                generate(self.request(WebInput()), [self.upload], self.settings)
            request = self.request(WebInput()).model_copy(update={'question_count': 100})
            with self.assertRaises(ValueError):
                generate(request, [], self.settings)
            llm.assert_not_called()

    def test_all_four_source_output_routes(self):
        page = Document(page_content='Evidence', metadata={'url': 'https://docs.python.org/types'})
        for document_mode in (True, False):
            for presentation in (True, False):
                source = DocumentInput(document_ids=['pending']) if document_mode else WebInput()
                request = (PresentationRequest(source=source, topic='Types', day=15,
                    audience='Engineers', template_id='reference') if presentation else self.request(source))
                names = ['get_llm', 'get_embeddings', 'get_search_tool', 'split_pages',
                         'build_document_index', 'build_web_index', 'build_retriever',
                         'build_interview_chains', 'build_presentation_chains', 'PDFInterviewPipeline',
                         'WebInterviewPipeline', 'PresentationPipeline', 'load_pdfs', 'research_web']
                from contextlib import ExitStack
                with ExitStack() as stack:
                    mocked = {name: stack.enter_context(patch('src.services.generation.'+name)) for name in names}
                    mocked['load_pdfs'].return_value = PDFLoadResult((page,), ('real-id',), ('warning',))
                    mocked['research_web'].return_value = WebLoadResult((page,), ())
                    mocked['split_pages'].return_value = [page] * 30
                    expected = 'PresentationPipeline' if presentation else 'PDFInterviewPipeline' if document_mode else 'WebInterviewPipeline'
                    mocked[expected].return_value.run.return_value = MagicMock()
                    mocked[expected].return_value.run.side_effect = [InsufficientEvidenceError('Too little context'),
                        mocked[expected].return_value.run.return_value]
                    result = generate(request, [self.upload] if document_mode else [], self.settings)
                    self.assertIs(result.output, mocked[expected].return_value.run.return_value)
                    routed = mocked[expected].return_value.run.call_args.args[0]
                    if document_mode:
                        self.assertEqual(routed.source.document_ids, ['real-id'])
                        mocked['research_web'].assert_not_called()
                        mocked['build_web_index'].assert_not_called()
                    else:
                        mocked['load_pdfs'].assert_not_called()
                        mocked['build_document_index'].assert_not_called()
                    self.assertEqual(request.source, source)  # caller's request remains unchanged
                    initial_k = self.settings.presentation_retrieval_k if presentation else self.settings.retrieval_k
                    self.assertEqual(mocked['build_retriever'].call_args_list[0].args[1].retrieval_k, initial_k)
                    self.assertEqual(mocked['build_retriever'].call_args.args[1].retrieval_k, initial_k * 2)
                    mocked['get_embeddings'].assert_called_once()
                    if not document_mode:
                        mocked['research_web'].assert_called_once()
                    # Recovery stays bounded when broader retrieval still cannot help.
                    mocked[expected].return_value.run.side_effect = [InsufficientEvidenceError('Still insufficient')] * 2
                    with self.assertRaises(InsufficientEvidenceError) as failure:
                        generate(request, [self.upload] if document_mode else [], self.settings)
                    self.assertEqual(failure.exception.recovery_attempts, 1)
                    self.assertEqual(failure.exception.requested_output, request.output)
                    self.assertEqual(mocked[expected].return_value.run.call_count, 4)


if __name__ == '__main__':
    unittest.main()
