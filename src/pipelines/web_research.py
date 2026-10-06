"""Plan searches, discover approved URLs, then fetch actual evidence pages."""

import json
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from src.core.config import Settings
from src.ingestion.web_loader import WebFetchError, WebLoadResult, fetch_page, validate_url
from src.ingestion.web_search import WebSearchError
from src.prompts.web import QUERY_PROMPT
from src.schemas.requests import InterviewRequest, PresentationRequest, WebInput
from src.schemas.web import SearchHit, SearchPlan


class WebResearchError(RuntimeError):
    """Not enough fetched evidence to proceed."""


def research_web(
    request: InterviewRequest | PresentationRequest, settings: Settings, llm: Runnable, search: Runnable,
    *, fetch: Callable[[str, Settings], Document] = fetch_page,
    progress: Callable[[str], None] = lambda message: None,
) -> WebLoadResult:
    if not isinstance(request.source, WebInput):
        raise ValueError('Web research cannot run in document mode.')
    parser = PydanticOutputParser(pydantic_object=SearchPlan)
    chain = QUERY_PROMPT.partial(count=settings.web_query_count,
        domains=json.dumps(settings.web_allowed_domains),
        format_instructions=parser.get_format_instructions()) | llm | parser
    plan = None
    progress('Planning online search queries')
    for _ in range(settings.interview_revision_limit + 1):
        try:
            candidate = chain.invoke({'request': request.model_dump_json()})
            if len(candidate.queries) == settings.web_query_count and len(set(candidate.queries)) == settings.web_query_count:
                plan = candidate
                break
        except (OutputParserException, ValidationError):
            continue
    if plan is None:
        raise WebResearchError('Search planner did not produce a valid bounded plan.')
    urls: dict[str, None] = {}
    warnings: list[str] = []
    for number, query in enumerate(plan.queries, 1):
        progress(f'Searching query {number} of {len(plan.queries)}')
        try:
            hits = search.invoke({'query': query})
            if not isinstance(hits, list):
                raise WebSearchError('Unexpected search response.')
            for item in hits[:settings.web_results_per_query]:
                try:
                    hit = SearchHit.model_validate(item)
                    url = validate_url(str(hit.url), settings.web_allowed_domains)
                    urls[url] = None
                except (ValidationError, WebFetchError):
                    warnings.append('Skipped invalid or unapproved search result.')
        except WebSearchError:
            warnings.append('One search failed; remaining queries continued.')
    pages: list[Document] = []
    fetched_urls: set[str] = set()
    total = 0
    # Bound fetch attempts as well as accepted pages.
    candidates = list(urls)[:settings.web_max_pages]
    for number, url in enumerate(candidates, 1):
        progress(f'Fetching source page {number} of {len(candidates)}')
        try:
            page = fetch(url, settings)
            final = validate_url(page.metadata['url'], settings.web_allowed_domains)
            page.metadata['url'] = final
            if final in fetched_urls:
                continue
            if total + len(page.page_content) > settings.web_max_total_chars:
                warnings.append('Skipped page exceeding the total text budget.')
                continue
            pages.append(page)
            fetched_urls.add(final)
            total += len(page.page_content)
            if page.metadata.get('truncated'):
                warnings.append('A fetched page was truncated to the configured text limit.')
        except WebFetchError:
            warnings.append('Skipped an inaccessible or unsupported page.')
    if len(pages) < settings.web_min_pages:
        raise WebResearchError('Too few approved pages fetched; check search configuration or provide more sources.')
    return WebLoadResult(tuple(pages), tuple(warnings))
