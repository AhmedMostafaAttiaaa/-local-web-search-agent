"""Text-stats tool: count words, characters, and lines of a piece of text.

A tiny offline helper for "how many words is this?" style questions, which
models otherwise answer by eyeballing (and get wrong).
"""
from __future__ import annotations

import json


def text_stats(text: str) -> str:
    """Return word/character/line counts for ``text`` as a JSON string.

    Never raises.

    Args:
        text: The text to measure.

    Returns:
        A JSON object string with ``words``, ``characters``,
        ``characters_no_spaces``, and ``lines``.
    """
    text = text or ""
    stats = {
        "words": len(text.split()),
        "characters": len(text),
        "characters_no_spaces": len("".join(text.split())),
        "lines": len(text.splitlines()) if text else 0,
    }
    return json.dumps(stats, ensure_ascii=False)
