"""Tests for the current_datetime tool (tools/datetime_tool.py)."""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.datetime_tool import current_datetime  # noqa: E402


def test_local_contains_today_and_label():
    out = current_datetime()
    assert "(local)" in out
    assert datetime.now().strftime("%Y-%m-%d") in out


def test_utc_label():
    assert "(UTC)" in current_datetime(utc=True)


def test_shape_matches_expected_format():
    # e.g. "Wednesday, 2026-08-06 14:30:05 (local)"
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \((local|UTC)\)", current_datetime())


if __name__ == "__main__":
    print(current_datetime())
    print(current_datetime(utc=True))
