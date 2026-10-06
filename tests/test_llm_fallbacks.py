"""Offline routing tests: no API keys, model clients, or network calls."""

import asyncio
import os
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.core.config import Settings
from src.core.llm_factory import AllProvidersUnavailableError, ContextLimitError, _compose


class QuotaError(Exception):
    status_code = 429


class AuthError(Exception):
    status_code = 401


class RequestError(Exception):
    status_code = 400


class FallbackTests(unittest.TestCase):
    def test_context_and_compatibility_rejections_try_other_models(self):
        for message in ('context_length_exceeded', 'model_decommissioned', 'unsupported response_format'):
            router, calls = self.router([RequestError(message), None, None])
            self.assertEqual(router.invoke('unchanged topic'), 'unchanged topic')
            self.assertEqual(calls, ['groq', 'openrouter'])

    def test_mixed_quota_context_failures_preserve_recovery_signal(self):
        for asynchronous in (False, True):
            router, calls = self.router([QuotaError(), RequestError('maximum context'), QuotaError()])
            with self.assertRaises(ContextLimitError):
                asyncio.run(router.ainvoke('topic')) if asynchronous else router.invoke('topic')
            self.assertEqual(len(calls), 3)

    def test_unknown_bad_request_is_not_hidden(self):
        router, calls = self.router([RequestError('invalid parameter'), None, None])
        with self.assertRaises(RequestError):
            router.invoke('topic')
        self.assertEqual(calls, ['groq'])

    def router(self, failures):
        calls = []
        providers = []
        for name, error in zip(("groq", "openrouter", "gemini"), failures):
            def invoke(value, name=name, error=error):
                calls.append(name)
                if error is not None:
                    raise error
                return value
            providers.append((name, RunnableLambda(invoke)))
        return _compose(providers), calls

    def test_third_provider_recovers(self):
        router, calls = self.router([QuotaError(), QuotaError(), None])
        self.assertEqual(router.invoke("context"), "context")
        self.assertEqual(calls, ["groq", "openrouter", "gemini"])

    def test_async_fallback(self):
        router, calls = self.router([QuotaError(), None, None])
        self.assertEqual(asyncio.run(router.ainvoke("context")), "context")
        self.assertEqual(calls, ["groq", "openrouter"])

    def test_primary_success_skips_fallbacks(self):
        router, calls = self.router([None, None, None])
        router.invoke("context")
        self.assertEqual(calls, ["groq"])

    def test_all_exhausted(self):
        router, _ = self.router([QuotaError(), QuotaError(), QuotaError()])
        with self.assertRaises(AllProvidersUnavailableError):
            router.invoke("context")

    def test_auth_and_programming_errors_propagate(self):
        for error in (AuthError(), ValueError("bad input")):
            router, calls = self.router([error, None, None])
            with self.assertRaises(type(error)):
                router.invoke("context")
            self.assertEqual(calls, ["groq"])

    def test_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            values = dict(
                _env_file=None, llm_provider="groq",
                llm_fallback_providers=("openrouter", "gemini"),
                groq_model="test", groq_api_key="test",
                openrouter_model="test:free", openrouter_api_key="test",
                gemini_model="test", google_api_key="test",
            )
            Settings(**values)
            with self.assertRaises(ValidationError):
                Settings(**{**values, "llm_fallback_providers": ("groq",)})
            with self.assertRaises(ValidationError):
                Settings(**{**values, "google_api_key": None})


if __name__ == "__main__":
    unittest.main()
