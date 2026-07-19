"""A tiny disk-backed TTL cache for search/fetch results.

Each CLI invocation is a fresh process, so an in-memory cache wouldn't help
across separate `python search_agent.py "..."` runs. Caching to disk lets
repeated queries within the TTL window skip the network entirely.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_CACHE_DIR = Path(".cache")
DEFAULT_TTL_SECONDS = 300  # 5 minutes


def _cache_path(cache_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def cached_call(
    key: str,
    fn: Callable[[], Any],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    should_cache: Callable[[Any], bool] = lambda value: True,
) -> Any:
    """Return the cached value for `key` if it's still fresh, else call `fn()`.

    On a cache miss (or expired/corrupt entry), `fn()` is called and returned.
    Its result is written to disk under `key` only if `should_cache(result)`
    is True (default: always) — e.g. pass ``lambda r: bool(r)`` to avoid
    caching empty/falsy results so a failed lookup is retried next time.
    """
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, key)

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            if time.time() - entry["cached_at"] <= ttl_seconds:
                return entry["value"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # treat as a miss

    value = fn()
    if should_cache(value):
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"cached_at": time.time(), "value": value}, fh)
    return value
