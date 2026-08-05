"""Tests for the text_stats tool (tools/text_stats.py)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.text_stats import text_stats  # noqa: E402


def test_counts_words_and_characters():
    stats = json.loads(text_stats("hello world"))
    assert stats["words"] == 2
    assert stats["characters"] == 11
    assert stats["characters_no_spaces"] == 10


def test_counts_lines():
    stats = json.loads(text_stats("a\nb\nc"))
    assert stats["lines"] == 3
    assert stats["words"] == 3


def test_empty_text():
    stats = json.loads(text_stats(""))
    assert stats == {"words": 0, "characters": 0, "characters_no_spaces": 0, "lines": 0}


if __name__ == "__main__":
    print(text_stats("the quick brown fox"))
