"""Validated LangChain retriever composition."""

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable, RunnableLambda

from src.core.config import IngestionSettings


def build_retriever(
    store: FAISS, settings: IngestionSettings | None = None,
) -> Runnable:
    """Return relevant Documents. MMR balances similarity with diversity.

    Use only a store built for the current authorized document selection.
    Nearest-neighbor results are candidates, not proof a question is answerable.
    """
    config = settings if settings is not None else IngestionSettings()

    def validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query must be a nonblank string.")
        query = query.strip()
        if len(query) > config.retrieval_max_query_chars:
            raise ValueError("Retrieval query exceeds the configured limit.")
        return query

    options = {"k": config.retrieval_k}
    if config.retrieval_search_type == "mmr":
        options.update(fetch_k=config.retrieval_fetch_k, lambda_mult=config.retrieval_mmr_lambda)
    return RunnableLambda(validate_query, name="validate_retrieval_query") | store.as_retriever(
        search_type=config.retrieval_search_type, search_kwargs=options
    )
