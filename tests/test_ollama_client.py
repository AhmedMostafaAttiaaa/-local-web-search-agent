"""Tests for the pure helper functions in agent/ollama_client.py.

These are the parsing/normalisation pieces of the tool-calling loop that need
no network access: message shaping, argument coercion, source extraction.
"""
from __future__ import annotations

import json

from agent.config import Config
from agent.ollama_client import (
    _coerce_args,
    _execute_tool,
    _extract_sources,
    _final_text,
    _message_to_dict,
    build_tool_schemas,
)


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
        cache_enabled=False,
        cache_ttl_seconds=300,
    )
    base.update(overrides)
    return Config(**base)


def test_build_tool_schemas_has_expected_tools():
    schemas = build_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"web_search", "fetch_page"}


def test_coerce_args_dict_passthrough():
    assert _coerce_args({"query": "x"}) == {"query": "x"}


def test_coerce_args_parses_json_string():
    assert _coerce_args('{"query": "x"}') == {"query": "x"}


def test_coerce_args_invalid_json_returns_empty():
    assert _coerce_args("not json") == {}


def test_coerce_args_non_dict_returns_empty():
    assert _coerce_args(42) == {}


def test_extract_sources_from_web_search_json():
    payload = json.dumps([{"title": "A", "url": "https://a"}, {"title": "B", "url": "https://b"}])
    assert _extract_sources(payload) == [
        {"title": "A", "url": "https://a"},
        {"title": "B", "url": "https://b"},
    ]


def test_extract_sources_ignores_malformed_json():
    assert _extract_sources("not json") == []


def test_extract_sources_ignores_non_list_json():
    assert _extract_sources(json.dumps({"not": "a list"})) == []


def test_final_text_prefers_content():
    assert _final_text({"content": "hello", "thinking": "ignored"}) == "hello"


def test_final_text_falls_back_to_thinking():
    assert _final_text({"content": "", "thinking": "reasoned answer"}) == "reasoned answer"


def test_final_text_empty_when_nothing_present():
    assert _final_text({}) == ""


def test_message_to_dict_normalises_tool_calls():
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
    }
    result = _message_to_dict(message)
    assert result["tool_calls"] == [{"function": {"name": "web_search", "arguments": {"query": "x"}}}]


def test_message_to_dict_passes_through_plain_message():
    message = {"role": "assistant", "content": "hi"}
    assert _message_to_dict(message) == {"role": "assistant", "content": "hi"}


def test_execute_tool_unknown_name():
    config = _make_config()
    assert _execute_tool("nope", {}, config) == "[error] Unknown tool: nope"


def test_execute_tool_web_search_missing_query():
    config = _make_config()
    assert _execute_tool("web_search", {}, config).startswith("[error]")


def test_execute_tool_fetch_page_missing_url():
    config = _make_config()
    assert _execute_tool("fetch_page", {}, config).startswith("[error]")
