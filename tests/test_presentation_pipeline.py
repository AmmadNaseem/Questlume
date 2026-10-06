"""Offline LCEL presentation generation, citation integrity and review checks."""

import json
import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from src.chains.presentation import build_presentation_chains
from src.core.config import Settings
from src.pipelines.interview import InsufficientEvidenceError
from src.pipelines.presentation import PresentationPipeline, PresentationQualityError
from src.schemas.requests import DocumentInput, PresentationRequest, WebInput


class PresentationTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.settings = Settings(_env_file=None, llm_provider='groq', llm_fallback_providers=(),
                groq_model='fake', groq_api_key='fake', presentation_revision_limit=1)
        self.request = PresentationRequest(topic='Type hints', day=15, audience='Python developers',
            template_id='reference', source=DocumentInput(document_ids=['d1']))
        self.chunk = Document(page_content='Type hints describe expected types.', metadata=dict(
            kind='document', source_id='s1', chunk_id='c1', document_id='d1', filename='notes.pdf', page=1))
        slides = [dict(number=1, layout='cover', purpose='Introduce topic', title='Type hints')]
        slides += [dict(number=i, layout='concept', purpose='Teach concept', title=f'Concept {i}',
                       bullets=['Type hints describe expected types.'], source_ids=['s1']) for i in range(2, 11)]
        self.draft = dict(sufficient_evidence=True, reason='Evidence supports the plan', slides=slides)
        self.accept = dict(approved=True, issues=[])

    def pipeline(self, responses, chunk=None, **kwargs):
        model = FakeListChatModel(responses=[json.dumps(value) if not isinstance(value, str) else value for value in responses])
        return PresentationPipeline(self.settings, RunnableLambda(lambda query: [chunk or self.chunk]),
                                    build_presentation_chains(model), **kwargs)

    def test_document_plan_uses_trusted_metadata(self):
        plan = self.pipeline([self.draft, self.accept]).run(self.request)
        self.assertEqual(len(plan.slides), 10)
        self.assertEqual((plan.day, plan.topic, plan.template_id), (15, 'Type hints', 'reference'))
        self.assertEqual(plan.sources[0].excerpt, self.chunk.page_content)

    def test_web_mode(self):
        chunk = Document(page_content=self.chunk.page_content, metadata=dict(kind='web', source_id='s1',
            chunk_id='c1', url='https://docs.python.org/types', title='Types'))
        request = self.request.model_copy(update={'source': WebInput()})
        plan = self.pipeline([self.draft, self.accept], chunk, web_urls=('https://docs.python.org/types',)).run(request)
        self.assertEqual(plan.source_mode, 'web')

    def test_parse_retry(self):
        self.assertEqual(len(self.pipeline(['invalid json', self.draft, self.accept]).run(self.request).slides), 10)

    def test_review_revision(self):
        reject = dict(approved=False, issues=['Unsupported example'])
        self.pipeline([self.draft, reject, self.draft, self.accept]).run(self.request)

    def test_never_force_approves(self):
        reject = dict(approved=False, issues=['Unsupported example'])
        with self.assertRaises(PresentationQualityError):
            self.pipeline([self.draft, reject]*2).run(self.request)

    def test_invalid_citations_count_order_and_content_limits(self):
        for mutation in ('citation', 'count', 'order', 'title', 'bullets', 'duplicates'):
            draft = json.loads(json.dumps(self.draft))
            if mutation == 'citation': draft['slides'][1]['source_ids'] = ['invented']
            elif mutation == 'count': draft['slides'].pop()
            elif mutation == 'order': draft['slides'][1]['number'] = 3
            elif mutation == 'title': draft['slides'][1]['title'] = 'a'*101
            elif mutation == 'bullets': draft['slides'][1]['bullets'] = ['x']*6
            else: draft['slides'][2]['title'] = draft['slides'][1]['title']
            with self.assertRaises(PresentationQualityError):
                self.pipeline([draft, draft]).run(self.request)

    def test_insufficient_evidence(self):
        with self.assertRaises(InsufficientEvidenceError):
            self.pipeline([dict(sufficient_evidence=False, reason='Not enough content', slides=[])]).run(self.request)

    def test_source_scope_and_budget(self):
        chunk = self.chunk.model_copy(deep=True)
        chunk.metadata['document_id'] = 'another'
        with self.assertRaises(ValueError):
            self.pipeline([self.draft], chunk).run(self.request)
        pipeline = self.pipeline([self.draft])
        pipeline.settings = self.settings.model_copy(update={'presentation_context_max_chars': 1})
        with self.assertRaises(InsufficientEvidenceError):
            pipeline.run(self.request)


if __name__ == '__main__':
    unittest.main()
