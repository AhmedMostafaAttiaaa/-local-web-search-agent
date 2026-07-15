"""Small HTTP helper: a verified request that adaptively falls back to an
unverified retry on TLS failure.

Why: some machines run a local HTTPS interceptor (corporate proxy or antivirus
such as Kaspersky) that re-signs traffic with a root CA Python doesn't trust.
The secure fix is the OS trust store via `truststore`, but that needs urllib3>=2
(Anaconda ships v1.x). To keep the tool WORKING everywhere while staying secure
where possible, this helper:

    1. tries the request with verification ON;
    2. only if that raises an SSLError, retries once with verification OFF
       (logging a single warning and muting urllib3's per-request noise).

On machines without interception, step 1 always succeeds, so verification stays
on and nothing is downgraded.
"""
from __future__ import annotations

import logging
from typing import Any, Union

import requests

logger = logging.getLogger(__name__)

# `verify` accepted by requests: True/False, or a path to a CA bundle.
VerifyType = Union[bool, str]

_insecure_warned = False


def _warn_insecure_once() -> None:
    """Log a single warning and silence urllib3's InsecureRequestWarning spam."""
    global _insecure_warned
    if not _insecure_warned:
        logger.warning(
            "TLS verification failed (likely a local HTTPS interceptor such as "
            "Kaspersky); retrying without verification. For secure verification, "
            "run with urllib3>=2 so the OS trust store (truststore) can be used."
        )
        _insecure_warned = True
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:  # noqa: BLE001 - best-effort noise suppression
        pass


def request(method: str, url: str, *, verify: VerifyType = True, **kwargs: Any) -> requests.Response:
    """Perform an HTTP request, retrying once without verification on SSL error.

    Args:
        method: HTTP method, e.g. "get" or "post".
        url: Target URL.
        verify: Initial verify value (True/False or CA bundle path).
        **kwargs: Passed through to requests (headers, params, data, timeout, ...).

    Returns:
        The requests.Response.

    Raises:
        requests.RequestException: If the request fails for a non-TLS reason, or
            if it still fails after the unverified retry.
    """
    try:
        return requests.request(method, url, verify=verify, **kwargs)
    except requests.exceptions.SSLError:
        if verify is False:
            raise  # already unverified; nothing more we can do
        _warn_insecure_once()
        return requests.request(method, url, verify=False, **kwargs)
