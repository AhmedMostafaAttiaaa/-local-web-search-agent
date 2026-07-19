"""Web search tools.

Primary backend:  SearxNG JSON API (self-hosted, free, no API key).
Fallback backend: DuckDuckGo HTML endpoint (scraped, no API key).

Every function returns a list of uniformly-shaped dicts::

    {"title": str, "url": str, "snippet": str}

The module is intentionally free of any agent/LLM logic so the search backend
can be swapped without touching the tool-calling loop.
"""
from __future__ import annotations

import logging
from typing import Any, Union
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from tools._http import request as http_request
from tools.cache import DEFAULT_TTL_SECONDS, cached_call

logger = logging.getLogger(__name__)

# `verify` accepted by requests: True/False, or a path to a CA bundle.
VerifyType = Union[bool, str]

# A realistic desktop User-Agent — some engines reject the default requests UA.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 10  # seconds — keep short so a dead backend fails fast.


def search_searxng(
    query: str,
    num_results: int = 5,
    base_url: str = "http://localhost:8080",
    verify: VerifyType = True,
) -> list[dict[str, str]]:
    """Search via a SearxNG instance's JSON API.

    Args:
        query: The search query.
        num_results: Maximum number of results to return.
        base_url: Base URL of the SearxNG instance.
        verify: Passed to requests' ``verify=`` (True/False or CA bundle path).

    Returns:
        A list of ``{"title", "url", "snippet"}`` dicts.

    Raises:
        requests.RequestException: On network/HTTP errors.
        ValueError: If the response is not valid JSON.
    """
    base_url = base_url.rstrip("/")
    params = {"q": query, "format": "json"}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    resp = http_request(
        "get",
        f"{base_url}/search",
        params=params,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        verify=verify,
    )
    resp.raise_for_status()

    try:
        data: dict[str, Any] = resp.json()
    except ValueError as exc:
        # Most common cause: JSON format not enabled in SearxNG settings.
        raise ValueError(
            "SearxNG did not return JSON (is `formats: [html, json]` enabled?)"
        ) from exc

    results: list[dict[str, str]] = []
    for item in data.get("results", []):
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": url,
                "snippet": (item.get("content") or "").strip(),
            }
        )
        if len(results) >= num_results:
            break
    return results


def _clean_ddg_url(href: str) -> str:
    """Resolve a DuckDuckGo redirect link (``/l/?uddg=...``) to the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return unquote(qs["uddg"][0])
    return href


def search_duckduckgo(
    query: str,
    num_results: int = 5,
    verify: VerifyType = True,
) -> list[dict[str, str]]:
    """Search by scraping DuckDuckGo's non-JS HTML endpoint (no API key).

    Args:
        query: The search query.
        num_results: Maximum number of results to return.
        verify: Passed to requests' ``verify=`` (True/False or CA bundle path).

    Returns:
        A list of ``{"title", "url", "snippet"}`` dicts.

    Raises:
        requests.RequestException: On network/HTTP errors.
    """
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": USER_AGENT}

    resp = http_request(
        "post", url, data={"q": query}, headers=headers, timeout=DEFAULT_TIMEOUT, verify=verify
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, str]] = []
    for block in soup.select(".result"):
        anchor = block.select_one("a.result__a")
        if not anchor:
            continue
        link = _clean_ddg_url(anchor.get("href", ""))
        if not link:
            continue
        snippet_el = block.select_one(".result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append(
            {
                "title": anchor.get_text(" ", strip=True),
                "url": link,
                "snippet": snippet,
            }
        )
        if len(results) >= num_results:
            break
    return results


def _web_search_uncached(
    query: str,
    num_results: int,
    prefer: str,
    searxng_host: str,
    verify: VerifyType,
) -> list[dict[str, str]]:
    """The actual multi-backend search, without any caching."""
    order = ["searxng", "duckduckgo"]
    if prefer == "duckduckgo":
        order = ["duckduckgo", "searxng"]

    last_error: Exception | None = None
    for backend in order:
        try:
            if backend == "searxng":
                results = search_searxng(query, num_results, searxng_host, verify=verify)
            else:
                results = search_duckduckgo(query, num_results, verify=verify)

            if results:
                logger.info("web_search: used %s (%d results)", backend, len(results))
                return results
            logger.warning("web_search: %s returned no results; trying next backend.", backend)
        except Exception as exc:  # noqa: BLE001 - we deliberately fall back on anything
            last_error = exc
            logger.warning("web_search: %s failed (%s); trying next backend.", backend, exc)

    logger.error("web_search: all backends failed. Last error: %s", last_error)
    return []


def web_search(
    query: str,
    num_results: int = 5,
    prefer: str = "searxng",
    searxng_host: str = "http://localhost:8080",
    verify: VerifyType = True,
    cache_enabled: bool = True,
    cache_ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> list[dict[str, str]]:
    """Search the web, trying the preferred backend first, then falling back.

    Tries SearxNG first (by default). If it raises OR returns zero results,
    falls back to DuckDuckGo. Logs which backend actually produced results.

    Args:
        query: The search query.
        num_results: Maximum number of results to return.
        prefer: ``"searxng"`` (default) or ``"duckduckgo"`` to try first.
        searxng_host: Base URL of the SearxNG instance.
        verify: Passed to requests' ``verify=`` (True/False or CA bundle path).
        cache_enabled: If True, cache results on disk for `cache_ttl_seconds`
            and reuse them for an identical query without hitting the network.
        cache_ttl_seconds: How long a cached result stays fresh.

    Returns:
        A list of ``{"title", "url", "snippet"}`` dicts. Empty list if every
        backend fails (never raises). Empty results are never cached, so a
        failed lookup is retried on the next call.
    """
    if not cache_enabled:
        return _web_search_uncached(query, num_results, prefer, searxng_host, verify)

    key = f"web_search:{prefer}:{searxng_host}:{num_results}:{query}"
    return cached_call(
        key,
        lambda: _web_search_uncached(query, num_results, prefer, searxng_host, verify),
        ttl_seconds=cache_ttl_seconds,
        should_cache=bool,
    )
