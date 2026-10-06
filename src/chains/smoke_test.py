"""First LCEL chain: validate input -> prompt -> model -> text parser."""

from collections.abc import Mapping
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from src.core.config import Settings, get_settings
from src.core.llm_factory import get_llm
from src.prompts.smoke_test import SMOKE_TEST_PROMPT


def build_smoke_test_chain(
    settings: Settings | None = None,
    *,
    llm: Runnable | None = None,
) -> Runnable:
    """Inject a fake model in tests; otherwise use configured provider routing."""
    config = settings if settings is not None else get_settings()

    def validate_input(value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("Chain input must be a mapping containing 'topic'.")
        topic = value.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic must be a nonblank string.")
        topic = topic.strip()
        if len(topic) > config.smoke_test_max_topic_chars:
            raise ValueError("Topic exceeds the configured length limit.")
        return {"topic": topic}

    model = llm if llm is not None else get_llm(config)
    prompt = SMOKE_TEST_PROMPT.partial(max_words=config.smoke_test_max_words)
    return (
        RunnableLambda(validate_input, name="validate_topic")
        | prompt
        | model
        | StrOutputParser()
    )
