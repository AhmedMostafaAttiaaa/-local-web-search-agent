"""Page-fetch tool: download a URL and extract clean, readable text.

Used by the agent to dig into a specific search result (e.g. to read an exact
price off a product page) beyond what the search snippet contains.
"""
from __future__ import annotations

import logging
import re
from typing import Union

import requests
from bs4 import BeautifulSoup

from tools._http import request as http_request

logger = logging.getLogger(__name__)

# `verify` accepted by requests: True/False, or a path to a CA bundle.
VerifyType = Union[bool, str]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 12  # seconds

# Structural / non-content tags we strip before extracting text.
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript")


def fetch_page(url: str, max_chars: int = 3000, verify: VerifyType = True) -> str:
    """Fetch ``url`` and return cleaned, whitespace-collapsed visible text.

    Never raises: on any error it returns a human-readable ``[fetch_page ...]``
    string so the agent can read the failure and react, instead of crashing.

    Args:
        url: The full URL to fetch (http/https).
        max_chars: Truncate the returned text to at most this many characters.
        verify: Passed to requests' ``verify=`` (True/False or CA bundle path).

    Returns:
        The cleaned page text, or a ``[fetch_page error]`` / ``[fetch_page notice]``
        message string.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = http_request("get", url, headers=headers, timeout=DEFAULT_TIMEOUT, verify=verify)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("fetch_page: request to %s failed: %s", url, exc)
        return f"[fetch_page error] Could not fetch {url}: {exc}"

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        logger.info("fetch_page: skipping non-HTML content (%s) at %s", content_type, url)
        return (
            f"[fetch_page notice] {url} returned non-HTML content "
            f"('{content_type or 'unknown'}'); no text extracted."
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    # get_text with a separator keeps words from different tags apart.
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return f"[fetch_page notice] No readable text extracted from {url}."

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ...[truncated]"

    return text
