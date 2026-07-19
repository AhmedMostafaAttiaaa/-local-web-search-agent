"""Tests for config loading precedence (env > yaml > defaults) and helpers."""
from __future__ import annotations

import pytest

from agent.config import Config, _as_bool, load_config

_ENV_VARS = ("OLLAMA_HOST", "OLLAMA_MODEL", "GROQ_MODEL", "GROQ_API_KEY", "SEARXNG_HOST", "REQUESTS_CA_BUNDLE")


@pytest.fixture(autouse=True)
def _clear_relevant_env(monkeypatch):
    # The project's own .env (loaded at import time via python-dotenv) would
    # otherwise leak OLLAMA_HOST/GROQ_API_KEY into these tests.
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_config(**overrides) -> Config:
    base = dict(
        backend="ollama",
        ollama_host="h",
        model="m",
        groq_model="g",
        groq_base_url="b",
        groq_api_key="",
        system_prompt="",
        searxng_host="s",
        num_results=5,
        auto_fetch_pages=False,
        max_page_chars=3000,
        verify_ssl=True,
        ca_bundle="",
        use_os_truststore=False,
        cache_enabled=True,
        cache_ttl_seconds=300,
    )
    base.update(overrides)
    return Config(**base)


def test_defaults_when_no_file(tmp_path):
    config = load_config(tmp_path / "no-such-config.yaml")
    assert config.backend == "ollama"
    assert config.model == "llama3.1"
    assert config.verify_ssl is True
    assert config.cache_enabled is True
    assert config.cache_ttl_seconds == 300


def test_yaml_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model: qwen3.6:35b\nnum_results: 8\n", encoding="utf-8")
    config = load_config(cfg_file)
    assert config.model == "qwen3.6:35b"
    assert config.num_results == 8


def test_env_overrides_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model: from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    config = load_config(cfg_file)
    assert config.model == "from-env"


def test_non_mapping_yaml_falls_back_to_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    config = load_config(cfg_file)
    assert config.model == "llama3.1"


def test_request_verify_prefers_ca_bundle():
    config = _make_config(verify_ssl=True, ca_bundle="/path/to/ca.pem")
    assert config.request_verify == "/path/to/ca.pem"


def test_request_verify_falls_back_to_verify_ssl():
    config = _make_config(verify_ssl=False, ca_bundle="")
    assert config.request_verify is False


@pytest.mark.parametrize(
    "value,expected",
    [(True, True), (False, False), ("true", True), ("YES", True), ("0", False), ("off", False), ("no", False)],
)
def test_as_bool(value, expected):
    assert _as_bool(value) is expected
