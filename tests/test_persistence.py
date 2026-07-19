"""Tests for --chat session persistence in agent/persistence.py."""
from __future__ import annotations

from agent.persistence import clear_history, load_history, save_history


def test_load_history_missing_session_returns_none(tmp_path):
    assert load_history("nope", sessions_dir=tmp_path) is None


def test_save_then_load_roundtrip(tmp_path):
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    save_history(history, name="mysession", sessions_dir=tmp_path)
    assert load_history("mysession", sessions_dir=tmp_path) == history


def test_save_overwrites_previous_history(tmp_path):
    save_history([{"role": "user", "content": "first"}], name="s", sessions_dir=tmp_path)
    save_history([{"role": "user", "content": "second"}], name="s", sessions_dir=tmp_path)
    assert load_history("s", sessions_dir=tmp_path) == [{"role": "user", "content": "second"}]


def test_clear_history_removes_file(tmp_path):
    save_history([{"role": "user", "content": "hi"}], name="s", sessions_dir=tmp_path)
    clear_history("s", sessions_dir=tmp_path)
    assert load_history("s", sessions_dir=tmp_path) is None


def test_clear_history_on_missing_session_is_a_noop(tmp_path):
    clear_history("never-existed", sessions_dir=tmp_path)  # must not raise


def test_corrupt_session_file_returns_none(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "default.json").write_text("not valid json", encoding="utf-8")
    assert load_history("default", sessions_dir=tmp_path) is None


def test_session_name_is_sanitised_and_cannot_escape_sessions_dir(tmp_path):
    save_history([{"role": "user", "content": "hi"}], name="weird/../name?", sessions_dir=tmp_path)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].parent == tmp_path
