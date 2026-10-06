"""Exercise full LCEL prompts, JSON parsers and orchestration offline."""

import json
import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from src.chains.interview import build_interview_chains
from src.core.config import Settings
from src.export.interview import render_interview_markdown
from src.pipelines.interview import InsufficientEvidenceError, InterviewQualityError, PDFInterviewPipeline
from src.schemas.requests import DocumentInput, InterviewRequest, WebInput


class InterviewPipelineTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.settings = Settings(_env_file=None, llm_provider='groq', llm_fallback_providers=(),
                groq_model='fake', groq_api_key='fake', interview_revision_limit=1)
        self.request = InterviewRequest(source=DocumentInput(document_ids=['d1']), topic='Type hints',
            job_role='Python developer', experience_level='junior', difficulty='beginner',
            question_count=1, question_types=['conceptual'])
        self.document = Document(page_content='Python type hints describe expected types, not runtime enforcement.',
            metadata=dict(kind='document', document_id='d1', filename='notes.pdf', page=1,
                          chunk_id='c1', source_id='c1'))
        self.question = dict(question='Do type hints enforce runtime types?', category='Python',
                             difficulty='beginner', question_type='conceptual', source_ids=['c1'])
        self.answer = dict(**self.question, answer='No. They describe expected types.',
                           key_points=['No runtime enforcement'], follow_up_questions=[], confidence='high')
        self.plan = dict(sufficient_evidence=True, reason='Evidence covers type hints', questions=[self.question])
        self.approved = dict(approved=True, issues=[])

    def pipeline(self, responses, document=None):
        model = FakeListChatModel(responses=[json.dumps(r) if not isinstance(r, str) else r for r in responses])
        retriever = RunnableLambda(lambda query: [document or self.document])
        return PDFInterviewPipeline(self.settings, retriever, build_interview_chains(model))

    def test_success_with_citations(self):
        result = self.pipeline([self.plan, self.answer, self.approved]).run(self.request)
        self.assertEqual(result.items[0].source_ids, ['c1'])
        self.assertEqual(result.sources[0].excerpt, self.document.page_content)
        rendered = render_interview_markdown(result)
        self.assertIn('notes.pdf, page 1', rendered)
        self.assertIn(self.answer['answer'], rendered)

    def test_parse_retry(self):
        result = self.pipeline(['not json', self.plan, self.answer, self.approved]).run(self.request)
        self.assertEqual(len(result.items), 1)

    def test_review_rejection_then_revision(self):
        reject = dict(approved=False, issues=['Unsupported answer'])
        result = self.pipeline([self.plan, self.answer, reject, self.plan, self.answer, self.approved]).run(self.request)
        self.assertEqual(len(result.items), 1)

    def test_never_force_approves(self):
        reject = dict(approved=False, issues=['Unsupported answer'])
        with self.assertRaises(InterviewQualityError):
            self.pipeline([self.plan, self.answer, reject]*2).run(self.request)

    def test_unknown_answer_citations(self):
        answer = {**self.answer, 'source_ids': ['invented']}
        with self.assertRaises(InterviewQualityError):
            self.pipeline([self.plan, answer]*2).run(self.request)

    def test_insufficient_evidence(self):
        with self.assertRaises(InsufficientEvidenceError):
            self.pipeline([dict(sufficient_evidence=False, reason='Not enough evidence', questions=[])]).run(self.request)

    def test_retriever_cannot_leak_other_documents(self):
        other = self.document.model_copy(deep=True)
        other.metadata['document_id'] = 'other'
        with self.assertRaises(ValueError):
            self.pipeline([self.plan], other).run(self.request)

    def test_request_mode_and_count_limits(self):
        for update in ({'source': WebInput()}, {'question_count': 11}):
            with self.assertRaises(ValueError):
                self.pipeline([self.plan]).run(self.request.model_copy(update=update))

    def test_context_budget(self):
        pipeline = self.pipeline([self.plan])
        pipeline.settings = self.settings.model_copy(update={'interview_context_max_chars': 1})
        with self.assertRaises(InsufficientEvidenceError):
            pipeline.run(self.request)


if __name__ == '__main__':
    unittest.main()
