"""Basic tests for the search + fetch tools.

These are light integration tests. The network-dependent ones (SearxNG /
DuckDuckGo / live page fetch) are skipped automatically when the backend is
unreachable, so the suite never hard-fails just because you're offline or
haven't started SearxNG.

Run with pytest:
    pytest -v

Or run this file directly for a quick manual smoke test:
    python tests/test_tools.py
"""
from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script (python tests/test_tools.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from tools.fetch_page import fetch_page  # noqa: E402
from tools.web_search import (  # noqa: E402
    _clean_ddg_url,
    search_duckduckgo,
    search_searxng,
    web_search,
)

SEARXNG_HOST = os.environ.get("SEARXNG_HOST", "http://localhost:8080")


def _assert_result_shape(results: list[dict]) -> None:
    """Every result must have exactly the title/url/snippet string keys."""
    assert isinstance(results, list)
    for item in results:
        assert set(item.keys()) == {"title", "url", "snippet"}
        assert isinstance(item["title"], str)
        assert isinstance(item["url"], str)
        assert isinstance(item["snippet"], str)


# --- pure/offline unit tests (always run) ------------------------------------

def test_clean_ddg_url_decodes_redirect() -> None:
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
    assert _clean_ddg_url(href) == "https://example.com/page"


def test_clean_ddg_url_passthrough() -> None:
    assert _clean_ddg_url("https://example.com/x") == "https://example.com/x"


def test_fetch_page_bad_url_returns_error_string() -> None:
    # Unroutable host -> must return an error string, never raise.
    out = fetch_page("http://localhost:0/nope", max_chars=200)
    assert isinstance(out, str)
    assert out.startswith("[fetch_page")


# --- network integration tests (skipped if backend unreachable) --------------

def test_search_duckduckgo_live() -> None:
    try:
        results = search_duckduckgo("python programming language", num_results=3)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DuckDuckGo unreachable: {exc}")
    _assert_result_shape(results)
    if not results:
        # DDG's HTML endpoint throttles automated requests and then returns an
        # empty page; treat that as "unavailable" rather than a hard failure.
        pytest.skip("DuckDuckGo returned no results (likely rate-limited).")


def test_search_searxng_live() -> None:
    try:
        results = search_searxng("openai", num_results=3, base_url=SEARXNG_HOST)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SearxNG unreachable at {SEARXNG_HOST}: {exc}")
    _assert_result_shape(results)


def test_web_search_falls_back_and_returns_shape() -> None:
    # Point SearxNG at a dead host to force the DuckDuckGo fallback path.
    try:
        results = web_search(
            "current weather london",
            num_results=3,
            prefer="searxng",
            searxng_host="http://127.0.0.1:1",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No usable search backend: {exc}")
    _assert_result_shape(results)


def test_fetch_page_live() -> None:
    try:
        text = fetch_page("https://example.com", max_chars=500)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"example.com unreachable: {exc}")
    assert isinstance(text, str)
    assert "example" in text.lower() or text.startswith("[fetch_page")


def _manual_smoke_test() -> None:
    """Human-readable smoke test when running this file directly."""
    print("== DuckDuckGo ==")
    try:
        for r in search_duckduckgo("price of iphone 16 in india", 3):
            print(f"  - {r['title']}\n    {r['url']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped: {exc})")

    print("\n== SearxNG @", SEARXNG_HOST, "==")
    try:
        for r in search_searxng("price of iphone 16 in india", 3, SEARXNG_HOST):
            print(f"  - {r['title']}\n    {r['url']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped: {exc})")

    print("\n== fetch_page(example.com) ==")
    print(" ", fetch_page("https://example.com", 300)[:300])


if __name__ == "__main__":
    _manual_smoke_test()
