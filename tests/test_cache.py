"""Tests for the disk-backed TTL cache in tools/cache.py."""
from __future__ import annotations

import time

from tools.cache import cached_call


def test_cached_call_reuses_fresh_value(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return "value"

    first = cached_call("key", compute, ttl_seconds=60, cache_dir=tmp_path)
    second = cached_call("key", compute, ttl_seconds=60, cache_dir=tmp_path)

    assert first == "value"
    assert second == "value"
    assert len(calls) == 1  # second call served from cache


def test_cached_call_recomputes_after_ttl_expiry(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return f"value-{len(calls)}"

    cached_call("key", compute, ttl_seconds=0, cache_dir=tmp_path)
    time.sleep(0.01)
    cached_call("key", compute, ttl_seconds=0, cache_dir=tmp_path)

    assert len(calls) == 2


def test_different_keys_do_not_collide(tmp_path):
    assert cached_call("a", lambda: "A", cache_dir=tmp_path) == "A"
    assert cached_call("b", lambda: "B", cache_dir=tmp_path) == "B"


def test_should_cache_false_skips_persisting(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return []  # falsy -> should_cache=bool means "don't persist"

    cached_call("key", compute, cache_dir=tmp_path, should_cache=bool)
    cached_call("key", compute, cache_dir=tmp_path, should_cache=bool)

    assert len(calls) == 2  # never served from cache since nothing was written


def test_corrupt_cache_file_is_treated_as_a_miss(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return "fresh"

    result = cached_call("key", compute, cache_dir=tmp_path)
    assert result == "fresh"

    # Corrupt the file that was just written.
    cache_files = list(tmp_path.iterdir())
    assert len(cache_files) == 1
    cache_files[0].write_text("not valid json", encoding="utf-8")

    result = cached_call("key", compute, cache_dir=tmp_path)
    assert result == "fresh"
    assert len(calls) == 2
