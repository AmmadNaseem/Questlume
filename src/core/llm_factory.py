"""Create provider-specific LangChain chat models from validated settings."""

import logging
from contextvars import ContextVar
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
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


class ContextLimitError(ProviderUnavailableError):
    """All attempted routing cannot fit the request; reduce evidence, not the topic."""


class ProviderOutputError(ProviderUnavailableError):
    """Model returned content that does not satisfy the requested contract."""


_context_failures: ContextVar[list[ContextLimitError] | None] = ContextVar('context_failures', default=None)


def _request_failure_kind(error: Exception) -> str | None:
    status = getattr(error, 'status_code', None) or getattr(error, 'code', None)
    if status is None:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
    # Inspect locally but never log provider bodies, which can contain prompt text.
    message = str(error).lower()
    if status == 413 or status in (400, 422) and any(marker in message for marker in (
            'context_length_exceeded', 'context window', 'maximum context', 'too many tokens',
            'request too large', 'reduce the length', 'input token limit')):
        return 'context'
    if status in (400, 404, 422) and any(marker in message for marker in (
            'model_not_found', 'model not found', 'model_decommissioned', 'model has been decommissioned',
            'model is not supported', 'unsupported model', 'response_format is not supported',
            'unsupported response_format', 'does not support json', 'no endpoints found')):
        return 'compatibility'
    return None


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


def _guard_provider(provider: str, model: Runnable, *, fallback: bool = False) -> Runnable:
    def normalize(error: Exception) -> None:
        if isinstance(error, OutputParserException):
            logger.warning('llm_output_rejected provider=%s reason=invalid_schema', provider)
            raise ProviderOutputError('Model returned invalid structured output.') from None
        kind = _request_failure_kind(error)
        if kind == 'context':
            logger.warning('llm_request_rejected provider=%s reason=context_limit', provider)
            failure = ContextLimitError('Provider context limit exceeded.')
            failures = _context_failures.get()
            if failures is not None:
                failures.append(failure)
            raise failure from None
        if kind == 'compatibility':
            logger.warning('llm_request_rejected provider=%s reason=model_compatibility', provider)
            raise ProviderUnavailableError('Configured model cannot handle this request.') from None
        if _is_availability_error(error):
            logger.warning("llm_provider_unavailable provider=%s", provider)
            raise ProviderUnavailableError(f"Provider unavailable: {provider}") from None

    def invoke(value: Any, config: RunnableConfig) -> Any:
        if fallback:
            logger.warning('llm_fallback_started provider=%s', provider)
        try:
            result = model.invoke(value, config=config)
            if fallback:
                logger.warning('llm_fallback_succeeded provider=%s', provider)
            return result
        except Exception as error:
            normalize(error)
            raise

    async def ainvoke(value: Any, config: RunnableConfig) -> Any:
        if fallback:
            logger.warning('llm_fallback_started provider=%s', provider)
        try:
            result = await model.ainvoke(value, config=config)
            if fallback:
                logger.warning('llm_fallback_succeeded provider=%s', provider)
            return result
        except Exception as error:
            normalize(error)
            raise

    return RunnableLambda(invoke, afunc=ainvoke, name=f"{provider}_guard")


def _compose(providers: list[tuple[str, Runnable]]) -> Runnable:
    guarded = [_guard_provider(name, model, fallback=index > 0)
               for index, (name, model) in enumerate(providers)]
    routed = guarded[0].with_fallbacks(
        guarded[1:], exceptions_to_handle=(ProviderUnavailableError,)
    )

    def invoke(value: Any, config: RunnableConfig) -> Any:
        failures: list[ContextLimitError] = []
        token = _context_failures.set(failures)
        try:
            return routed.invoke(value, config=config)
        except ContextLimitError:
            raise
        except ProviderOutputError:
            if failures:
                raise failures[0] from None
            raise OutputParserException('Configured models returned invalid structured output.') from None
        except ProviderUnavailableError:
            if failures:
                raise failures[0] from None
            raise AllProvidersUnavailableError(
                "All configured providers are unavailable. Please retry later."
            ) from None
        finally:
            _context_failures.reset(token)

    async def ainvoke(value: Any, config: RunnableConfig) -> Any:
        failures: list[ContextLimitError] = []
        token = _context_failures.set(failures)
        try:
            return await routed.ainvoke(value, config=config)
        except ContextLimitError:
            raise
        except ProviderOutputError:
            if failures:
                raise failures[0] from None
            raise OutputParserException('Configured models returned invalid structured output.') from None
        except ProviderUnavailableError:
            if failures:
                raise failures[0] from None
            raise AllProvidersUnavailableError(
                "All configured providers are unavailable. Please retry later."
            ) from None
        finally:
            _context_failures.reset(token)

    return RunnableLambda(invoke, afunc=ainvoke, name="llm_fallback_router")


def get_llm(
    settings: Settings | None = None, *, schema: type[BaseModel] | None = None,
    parsed_schema: type[BaseModel] | None = None,
) -> Runnable:
    """Create once at startup; return a sync/async LangChain fallback Runnable.

    Structured output is applied to each model before composing fallbacks.
    Authentication, unrecognized invalid requests, and parsing failures are not hidden.
    """
    config = settings if settings is not None else get_settings()
    if schema is not None and parsed_schema is not None:
        raise ValueError('Choose either native structured output or parsed output.')
    models: list[tuple[str, Runnable]] = []
    for provider in (config.llm_provider, *config.llm_fallback_providers):
        model = _build_model(provider, config)
        runnable = model.with_structured_output(schema) if schema is not None else model
        if parsed_schema is not None:
            runnable = runnable | PydanticOutputParser(pydantic_object=parsed_schema)
        models.append((provider, runnable))
    return _compose(models)
