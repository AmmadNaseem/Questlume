"""Real FAISS tests with deterministic offline embeddings."""

import asyncio
import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from src.core.config import IngestionSettings
from src.retrieval.chunking import split_pages
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import build_document_index


class TopicEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_texts = []

    def embed_query(self, text):
        return [float(text.lower().count('python')), float(text.lower().count('angular')), 0.1]

    def embed_documents(self, texts):
        self.embedded_texts.extend(texts)
        return [self.embed_query(text) for text in texts]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.settings = IngestionSettings(_env_file=None, retrieval_k=1, retrieval_fetch_k=3)
        pages = [Document(page_content=text, metadata=dict(
            kind='document', document_id=identity, filename=identity+'.pdf', page=1,
        )) for identity, text in [('python', 'Python type hints'), ('angular', 'Angular components')]]
        self.chunks = split_pages(pages, self.settings)
        self.embeddings = TopicEmbeddings()

    def test_selection_before_embedding(self):
        store = build_document_index(self.chunks, ['python'], self.embeddings)
        docs = build_retriever(store, self.settings).invoke('Angular')
        self.assertEqual(docs[0].metadata['document_id'], 'python')
        self.assertEqual(self.embeddings.embedded_texts, ['Python type hints'])

    def test_similarity_and_mmr_sync_async(self):
        store = build_document_index(self.chunks, ['python', 'angular'], self.embeddings)
        for search_type in ('similarity', 'mmr'):
            config = self.settings.model_copy(update={'retrieval_search_type': search_type})
            retriever = build_retriever(store, config)
            self.assertEqual(retriever.invoke('Python')[0].metadata['document_id'], 'python')
            self.assertEqual(asyncio.run(retriever.ainvoke('Angular'))[0].metadata['document_id'], 'angular')

    def test_selection_errors(self):
        for ids in ([], ['unknown'], ['python', 'python'], [' '], 'python'):
            with self.assertRaises(ValueError):
                build_document_index(self.chunks, ids, self.embeddings)
        self.assertEqual(self.embeddings.embedded_texts, [])

    def test_invalid_queries(self):
        store = build_document_index(self.chunks, ['python'], self.embeddings)
        retriever = build_retriever(store, self.settings)
        for query in (' ', None, 'a'*2001):
            with self.assertRaises(ValueError):
                retriever.invoke(query)

    def test_store_copies_provenance(self):
        store = build_document_index(self.chunks, ['python'], self.embeddings)
        self.chunks[0].metadata['page'] = 999
        self.assertEqual(build_retriever(store, self.settings).invoke('Python')[0].metadata['page'], 1)

    def test_invalid_fetch_limit(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValidationError):
            IngestionSettings(_env_file=None, retrieval_k=5, retrieval_fetch_k=2)


if __name__ == '__main__':
    unittest.main()
