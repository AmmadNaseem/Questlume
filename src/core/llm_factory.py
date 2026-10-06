"""Create provider-specific LangChain chat models from validated settings."""

import logging
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from src.core.config import Settings, get_settings


def _build_model(provider: str, config: Settings) -> BaseChatModel:
    """Build the configured chat model without generating any content.

    Pass explicit settings in tests or at application startup.
    Otherwise, use the cached application settings.
    """

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key.get_secret_value(),
            temperature=config.llm_temperature,
            timeout=config.llm_timeout_seconds,
            max_retries=config.llm_max_retries,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY is required.")

        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            api_key=config.google_api_key.get_secret_value(),
            vertexai=False,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout_seconds,
            max_retries=config.llm_max_retries,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        if config.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required.")

        return ChatOpenRouter(
            model=config.openrouter_model,
            api_key=config.openrouter_api_key.get_secret_value(),
            temperature=config.llm_temperature,
            timeout=int(config.llm_timeout_seconds * 1_000),
            max_retries=config.llm_max_retries,
        )

    raise ValueError("Unsupported LLM provider.")


logger = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    """Provider quota or temporary availability failure."""


class AllProvidersUnavailableError(RuntimeError):
    """All providers failed; callers should offer a retry later."""


def _is_availability_error(error: Exception) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (httpx.TransportError, TimeoutError, ConnectionError)):
            return True
        status = getattr(current, "status_code", None)
        if status is None:
            status = getattr(current, "code", None)
        if status is None:
            status = getattr(getattr(current, "response", None), "status_code", None)
        if isinstance(status, int) and (status in (402, 408, 429) or 500 <= status < 600):
            return True
        current = current.__cause__ or current.__context__
    return False


def _guard_provider(provider: str, model: Runnable) -> Runnable:
    def normalize(error: Exception) -> None:
        if _is_availability_error(error):
            logger.warning("llm_provider_unavailable provider=%s", provider)
            raise ProviderUnavailableError(f"Provider unavailable: {provider}") from None

    def invoke(value: Any, config: RunnableConfig) -> Any:
        try:
            return model.invoke(value, config=config)
        except Exception as error:
            normalize(error)
            raise

    async def ainvoke(value: Any, config: RunnableConfig) -> Any:
        try:
            return await model.ainvoke(value, config=config)
        except Exception as error:
            normalize(error)
            raise

    return RunnableLambda(invoke, afunc=ainvoke, name=f"{provider}_guard")


def _compose(providers: list[tuple[str, Runnable]]) -> Runnable:
    guarded = [_guard_provider(name, model) for name, model in providers]
    routed = guarded[0].with_fallbacks(
        guarded[1:], exceptions_to_handle=(ProviderUnavailableError,)
    )

    def invoke(value: Any, config: RunnableConfig) -> Any:
        try:
            return routed.invoke(value, config=config)
        except ProviderUnavailableError:
            raise AllProvidersUnavailableError(
                "All configured providers are unavailable. Please retry later."
            ) from None

    async def ainvoke(value: Any, config: RunnableConfig) -> Any:
        try:
            return await routed.ainvoke(value, config=config)
        except ProviderUnavailableError:
            raise AllProvidersUnavailableError(
                "All configured providers are unavailable. Please retry later."
            ) from None

    return RunnableLambda(invoke, afunc=ainvoke, name="llm_fallback_router")


def get_llm(
    settings: Settings | None = None, *, schema: type[BaseModel] | None = None
) -> Runnable:
    """Create once at startup; return a sync/async LangChain fallback Runnable.

    Structured output is applied to each model before composing fallbacks.
    Authentication, invalid requests, and parsing failures are not hidden.
    """
    config = settings if settings is not None else get_settings()
    models: list[tuple[str, Runnable]] = []
    for provider in (config.llm_provider, *config.llm_fallback_providers):
        model = _build_model(provider, config)
        runnable = model.with_structured_output(schema) if schema is not None else model
        models.append((provider, runnable))
    return _compose(models)
