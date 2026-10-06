"""Configurable search APIs wrapped in a LangChain StructuredTool."""

import httpx
from langchain_core.tools import StructuredTool

from src.core.config import Settings
from src.schemas.web import SearchHit


class WebSearchError(RuntimeError):
    """Search credentials, quota, response, or connectivity failed."""


def get_search_tool(settings: Settings) -> StructuredTool:
    provider = settings.web_search_provider
    key = settings.tavily_api_key if provider == "tavily" else settings.google_search_api_key
    if key is None or not key.get_secret_value().strip():
        raise WebSearchError(f"Configure the {provider} search API key.")
    if provider == "google" and not settings.google_cse_id.strip():
        raise WebSearchError("Configure GOOGLE_CSE_ID.")

    def search(query: str) -> list[dict]:
        """Discover page URLs; snippets are not used as generation evidence."""
        if not query.strip() or len(query) > settings.retrieval_max_query_chars:
            raise WebSearchError("Invalid search query.")
        try:
            with httpx.Client(timeout=settings.web_timeout_seconds, trust_env=False) as client:
                if provider == "tavily":
                    response = client.post('https://api.tavily.com/search', json={
                        'api_key': key.get_secret_value(), 'query': query,
                        'max_results': settings.web_results_per_query,
                        'include_domains': list(settings.web_allowed_domains),
                        'search_depth': 'basic', 'include_raw_content': False,
                    })
                    response.raise_for_status()
                    items = response.json().get('results', [])
                    hits = [SearchHit(url=item['url'], title=item.get('title') or item['url']) for item in items]
                else:
                    response = client.get('https://www.googleapis.com/customsearch/v1', params={
                        'key': key.get_secret_value(), 'cx': settings.google_cse_id,
                        'q': query, 'num': settings.web_results_per_query,
                    })
                    response.raise_for_status()
                    hits = [SearchHit(url=item['link'], title=item['title']) for item in response.json().get('items', [])]
                return [hit.model_dump(mode='json') for hit in hits]
        except Exception:
            raise WebSearchError("Search request failed; check credentials, quota and connectivity.") from None

    return StructuredTool.from_function(search, name='web_search', description=search.__doc__)
