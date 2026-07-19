"""Persist `--chat` conversation history to disk so it survives across runs.

Each named session is a JSON file holding the message list `run_agent()`
already passes around as `history`. Resuming just means loading that list and
handing it back to `run_agent()` as the starting history.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SESSIONS_DIR = Path(".chat_sessions")
DEFAULT_SESSION_NAME = "default"


def _session_path(name: str, sessions_dir: Path | str = DEFAULT_SESSIONS_DIR) -> Path:
    """Map a (possibly untrusted) session name to a safe file under sessions_dir."""
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or DEFAULT_SESSION_NAME
    return Path(sessions_dir) / f"{safe_name}.json"


def load_history(
    name: str = DEFAULT_SESSION_NAME, sessions_dir: Path | str = DEFAULT_SESSIONS_DIR
) -> list[dict[str, Any]] | None:
    """Return the saved message history for `name`, or None if absent/invalid."""
    path = _session_path(name, sessions_dir)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, list) else None


def save_history(
    history: list[dict[str, Any]],
    name: str = DEFAULT_SESSION_NAME,
    sessions_dir: Path | str = DEFAULT_SESSIONS_DIR,
) -> None:
    """Persist `history` for `name`, overwriting any previous save."""
    sessions_dir = Path(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = _session_path(name, sessions_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)


def clear_history(name: str = DEFAULT_SESSION_NAME, sessions_dir: Path | str = DEFAULT_SESSIONS_DIR) -> None:
    """Delete the saved history for `name`, if any."""
    _session_path(name, sessions_dir).unlink(missing_ok=True)
