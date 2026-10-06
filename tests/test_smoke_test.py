"""Check real LCEL composition offline with a fake chat model."""

import asyncio
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.chains.smoke_test import build_smoke_test_chain
from src.core.config import Settings
from src.main import main


class SmokeTestTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.settings = Settings(
                _env_file=None, llm_provider="groq", llm_fallback_providers=(),
                groq_model="test", groq_api_key="test",
                smoke_test_max_topic_chars=30,
            )
        self.prompts = []

        def model(prompt):
            self.prompts.append(prompt.to_messages())
            return AIMessage(content="Parsed explanation")

        self.chain = build_smoke_test_chain(self.settings, llm=RunnableLambda(model))

    def test_prompt_and_parser(self):
        self.assertEqual(self.chain.invoke({"topic": "  Python {types}  "}), "Parsed explanation")
        self.assertIn("120", self.prompts[0][0].content)
        self.assertEqual(self.prompts[0][1].content, "Topic: Python {types}")

    def test_async(self):
        self.assertEqual(asyncio.run(self.chain.ainvoke({"topic": "Python"})), "Parsed explanation")

    def test_invalid_input_never_calls_model(self):
        for value in ({"topic": " "}, {}, {"topic": 3}, {"topic": "a" * 31}, "Python"):
            with self.assertRaises(ValueError):
                self.chain.invoke(value)
        self.assertEqual(self.prompts, [])

    def test_offline_cli(self):
        with patch.dict(os.environ, {}, clear=True), patch("builtins.print") as output:
            self.assertEqual(main(["--topic", "Python", "--offline"]), 0)
            self.assertIn("Offline smoke test succeeded", output.call_args.args[0])

    def test_cli_rejects_blank_topic(self):
        with patch.dict(os.environ, {}, clear=True), patch("builtins.print"):
            self.assertEqual(main(["--topic", " ", "--offline"]), 2)


if __name__ == "__main__":
    unittest.main()
