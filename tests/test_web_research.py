"""Offline web tests; fake search, HTTP transport, DNS and LLM outputs."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from src.chains.interview import build_interview_chains
from src.core.config import Settings
from src.ingestion.web_loader import WebFetchError, _public_address, fetch_page, validate_url
from src.ingestion.web_search import WebSearchError, get_search_tool
from src.pipelines.interview import WebInterviewPipeline
from src.pipelines.web_research import WebResearchError, research_web
from src.retrieval.chunking import split_pages
from src.schemas.requests import InterviewRequest, WebInput


class WebTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.settings = Settings(_env_file=None, llm_provider='groq', llm_fallback_providers=(),
                groq_model='fake', groq_api_key='fake', tavily_api_key='fake',
                web_allowed_domains=('docs.python.org',), web_query_count=2,
                web_min_pages=1, interview_revision_limit=0)
        self.request = InterviewRequest(source=WebInput(), topic='Python types', job_role='Developer',
            experience_level='junior', difficulty='beginner', question_count=1, question_types=['conceptual'])

    def test_url_restrictions(self):
        for url in ('http://docs.python.org/', 'https://docs.python.org.evil.test/',
                    'https://user:pass@docs.python.org/', 'https://127.0.0.1/',
                    'https://docs.python.org:1234/'):
            with self.assertRaises(WebFetchError):
                validate_url(url, self.settings.web_allowed_domains)
        self.assertEqual(validate_url('https://docs.python.org/a#section', self.settings.web_allowed_domains),
                         'https://docs.python.org/a')

    def test_private_dns_rejected(self):
        with patch('socket.getaddrinfo', return_value=[(None, None, None, None, ('127.0.0.1', 443))]):
            with self.assertRaises(WebFetchError):
                _public_address('docs.python.org')

    def fetch_mock(self, handler, settings=None):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch('src.ingestion.web_loader.httpx.Client', return_value=client), \
             patch('src.ingestion.web_loader._public_address', return_value='8.8.8.8'):
            return fetch_page('https://docs.python.org/types', settings or self.settings)

    def test_cleaning_and_pinned_destination(self):
        def handler(request):
            self.assertEqual(request.url.host, '8.8.8.8')
            self.assertEqual(request.headers['host'], 'docs.python.org')
            self.assertEqual(request.extensions['sni_hostname'], 'docs.python.org')
            return httpx.Response(200, headers={'content-type':'text/html'},
                text='<html><title>Types</title><nav>Navigation</nav><main><script>bad()</script><p>Python types.</p></main></html>')
        page = self.fetch_mock(handler)
        self.assertEqual(page.page_content, 'Python types.')
        self.assertEqual(page.metadata['url'], 'https://docs.python.org/types')

    def test_redirect_and_size_limits(self):
        with self.assertRaises(WebFetchError):
            self.fetch_mock(lambda request: httpx.Response(302, headers={'location':'https://127.0.0.1/'}))
        with self.assertRaises(WebFetchError):
            self.fetch_mock(lambda request: httpx.Response(200, headers={'content-type':'text/html'}, text='long body'),
                            self.settings.model_copy(update={'web_max_page_bytes': 1}))

    def test_search_tool(self):
        def handler(request):
            self.assertEqual(request.url.host, 'api.tavily.com')
            return httpx.Response(200, json={'results':[{'url':'https://docs.python.org/types', 'title':'Types', 'content':'snippet'}]})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch('src.ingestion.web_search.httpx.Client', return_value=client):
            result = get_search_tool(self.settings).invoke({'query':'Python types'})
        self.assertNotIn('content', result[0])

    def test_search_key_checked_only_for_web(self):
        with self.assertRaises(WebSearchError):
            get_search_tool(self.settings.model_copy(update={'tavily_api_key': None}))

    def test_research_deduplicates_and_uses_fetched_content(self):
        llm = FakeListChatModel(responses=[json.dumps({'queries':['types docs', 'types examples']})])
        search = RunnableLambda(lambda value: [dict(url='https://docs.python.org/types', title='Types')])
        calls = []
        def fetch(url, config):
            calls.append(url)
            return Document(page_content='Fetched evidence', metadata=dict(kind='web', url=url, title='Types'))
        result = research_web(self.request, self.settings, llm, search, fetch=fetch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.pages[0].page_content, 'Fetched evidence')

    def test_research_requires_minimum_pages(self):
        llm = FakeListChatModel(responses=[json.dumps({'queries':['one', 'two']})])
        with self.assertRaises(WebResearchError):
            research_web(self.request, self.settings, llm, RunnableLambda(lambda value: []))

    def test_web_interview_citations(self):
        page = Document(page_content='Python type hints describe types.',
                        metadata=dict(kind='web', url='https://docs.python.org/types', title='Types'))
        chunk = split_pages([page], self.settings)[0]
        source_id = chunk.metadata['source_id']
        question = dict(question='What do type hints describe?', category='Python', difficulty='beginner',
                        question_type='conceptual', source_ids=[source_id])
        answer = dict(**question, answer='Expected types.', key_points=['Types'], confidence='high')
        responses = [dict(sufficient_evidence=True, reason='Supported', questions=[question]), answer,
                     dict(approved=True, issues=[])]
        model = FakeListChatModel(responses=[json.dumps(value) for value in responses])
        pipeline = WebInterviewPipeline(self.settings, RunnableLambda(lambda query: [chunk]),
            build_interview_chains(model), ['https://docs.python.org/types'])
        result = pipeline.run(self.request)
        self.assertEqual(result.source_mode, 'web')
        self.assertEqual(str(result.sources[0].url), 'https://docs.python.org/types')


if __name__ == '__main__':
    unittest.main()
