"""Date/time tool: tell the agent the current date and time (offline).

Models have no clock and often guess "today" wrong, which breaks time-sensitive
reasoning ("latest", "this year", age calculations). This returns the real
current date/time from the machine, using only the standard library.
"""
from __future__ import annotations

from datetime import datetime, timezone


def current_datetime(utc: bool = False) -> str:
    """Return the current date and time as a readable string.

    Never raises.

    Args:
        utc: If True, return UTC; otherwise the machine's local time.

    Returns:
        A string like ``"Wednesday, 2026-08-06 14:30:05 (local)"``.
    """
    now = datetime.now(timezone.utc) if utc else datetime.now()
    label = "UTC" if utc else "local"
    return f"{now.strftime('%A, %Y-%m-%d %H:%M:%S')} ({label})"
