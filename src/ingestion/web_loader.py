"""Bounded HTTPS fetching from approved domains, pinned to public DNS addresses."""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.core.config import Settings


class WebFetchError(ValueError):
    """A page is unsafe, inaccessible, unsupported, or has no usable text."""


def validate_url(url: str, domains: tuple[str, ...]) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or '').lower()
    try:
        port = parsed.port
    except ValueError:
        raise WebFetchError('Invalid URL port.') from None
    if parsed.scheme != 'https' or port not in (None, 443) or parsed.username or parsed.password:
        raise WebFetchError('Only HTTPS URLs without credentials on port 443 are allowed.')
    if not host or not any(host == domain.lower() or host.endswith('.' + domain.lower()) for domain in domains):
        raise WebFetchError('URL is outside the configured approved domains.')
    return urlunsplit(('https', host, parsed.path or '/', parsed.query, ''))


def _public_address(host: str) -> str:
    try:
        addresses = [entry[4][0] for entry in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)]
    except OSError:
        raise WebFetchError('DNS resolution failed.') from None
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise WebFetchError('Nonpublic destination addresses are forbidden.')
    return addresses[0]


def fetch_page(url: str, settings: Settings) -> Document:
    current = validate_url(url, settings.web_allowed_domains)
    try:
        with httpx.Client(timeout=settings.web_timeout_seconds, follow_redirects=False, trust_env=False) as client:
            for redirect in range(settings.web_max_redirects + 1):
                host = urlsplit(current).hostname
                address = _public_address(host)
                # Connect to the vetted IP, preserving Host and TLS certificate/SNI checks.
                target = httpx.URL(current).copy_with(host=address)
                with client.stream('GET', target, headers={'Host': host},
                                   extensions={'sni_hostname': host}) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirect == settings.web_max_redirects or 'location' not in response.headers:
                            raise WebFetchError('Redirect limit reached or missing redirect location.')
                        current = validate_url(urljoin(current, response.headers['location']), settings.web_allowed_domains)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get('content-type', '').split(';')[0].strip().lower()
                    if content_type not in ('text/html', 'application/xhtml+xml', 'text/plain'):
                        raise WebFetchError('Unsupported page content type.')
                    body = bytearray()
                    for block in response.iter_bytes():
                        body.extend(block)
                        if len(body) > settings.web_max_page_bytes:
                            raise WebFetchError('Page exceeds the configured download limit.')
                    html = bytes(body).decode(response.encoding or 'utf-8', errors='replace')
                    if content_type == 'text/plain':
                        text, title = html.strip(), current
                    else:
                        soup = BeautifulSoup(html, 'html.parser')
                        title = soup.title.get_text(' ', strip=True) if soup.title else current
                        for node in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg', 'form']):
                            node.decompose()
                        content = soup.find('main') or soup.find('article') or soup.body or soup
                        text = '\n'.join(line.strip() for line in content.get_text('\n').splitlines() if line.strip())
                    if not text:
                        raise WebFetchError('Page contains no usable text.')
                    # Explicit bounded excerpt; metadata records truncation.
                    return Document(page_content=text[:settings.web_max_page_chars], metadata={
                        'kind': 'web', 'url': current, 'title': title or current,
                        'truncated': len(text) > settings.web_max_page_chars,
                    })
    except WebFetchError:
        raise
    except Exception:
        raise WebFetchError('Page fetch failed.') from None
    raise WebFetchError('Page fetch did not produce a document.')


@dataclass(frozen=True)
class WebLoadResult:
    pages: tuple[Document, ...]
    warnings: tuple[str, ...]
