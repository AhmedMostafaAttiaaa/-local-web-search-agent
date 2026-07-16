"""The Ollama tool-calling agent loop.

This module owns the conversation loop that lets an Ollama model call our local
tools (`web_search`, `fetch_page`) and reason over their results.

THE TOOL-CALLING FLOW (read this to understand the pattern)
-----------------------------------------------------------
1. We send the user's message + a list of TOOL SCHEMAS (JSON descriptions of
   each tool) to the model via `ollama.Client.chat(..., tools=...)`.
2. The model replies with EITHER plain text (it's done) OR a `tool_calls`
   list — structured requests like ``web_search(query="...")``.
3. For each requested tool call, WE (not the model) execute the real Python
   function locally and capture its output.
4. We append the model's assistant message AND a ``role: "tool"`` message
   (carrying the output) back into the conversation, then call the model again.
5. The model now sees the tool results and either calls more tools or writes a
   final answer. We repeat until it stops calling tools (or we hit a cap).

The model decides *when* and *how* to call tools; this loop just wires the
requests to real functions and feeds results back.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import ollama

from agent.config import Config, enable_os_truststore, load_config
from tools.fetch_page import fetch_page
from tools.web_search import web_search

logger = logging.getLogger(__name__)

# --- optional pretty output ---------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel

    _console: Console | None = Console()
except ImportError:  # pragma: no cover - rich is optional
    _console = None


SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to live web-search tools.\n"
    "When a question depends on current or real-time facts (prices, exchange "
    "rates, news, releases), use `web_search` rather than guessing.\n"
    "IMPORTANT: each search result includes a snippet, and these snippets very "
    "often already contain the answer (e.g. an exchange rate or price). Read the "
    "snippets first and answer directly from them whenever possible.\n"
    "Only use `fetch_page` if the snippets are genuinely insufficient. If a page "
    "returns an error (403/404) or its text does not contain the figure, do NOT "
    "keep retrying pages — fall back to the numbers already shown in the search "
    "snippets.\n"
    "Always finish with a short, direct answer that states the concrete figure, "
    "then cite the source URLs. Do not narrate your process or your uncertainty."
)


def list_models(host: str) -> list[str]:
    """Return the model names available on an Ollama host (empty list on error).

    Works whether the client returns dicts or pydantic objects.
    """
    try:
        data = ollama.Client(host=host).list()
    except Exception:  # noqa: BLE001 - best-effort; used only for nicer errors
        return []
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names: list[str] = []
    for item in models or []:
        name = item.get("model") if isinstance(item, dict) else getattr(item, "model", None)
        if name:
            names.append(name)
    return names


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return the OpenAI-style function schemas Ollama expects in `tools`."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the live internet and return a list of results, each "
                    "with a title, url, and snippet. Use this whenever the answer "
                    "may depend on current or real-time information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "How many results to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": (
                    "Fetch a single web page by URL and return its cleaned, "
                    "readable text. Use after web_search to read a specific "
                    "result in detail (e.g. to find an exact price)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to fetch.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Max characters of text to return.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def _print_tool_call(step: int, name: str, args: dict[str, Any]) -> None:
    """Show the agent's reasoning: which tool it called and with what args."""
    arg_str = json.dumps(args, ensure_ascii=False)
    if _console is not None:
        _console.print(
            Panel.fit(
                f"[bold cyan]{name}[/bold cyan]([white]{arg_str}[/white])",
                title=f"Tool call - step {step}",
                border_style="cyan",
            )
        )
    else:
        print(f"[step {step}] -> {name}({arg_str})")


def _chat_with_status(client: Any, model: str, messages: list, tools: Any, label: str) -> Any:
    """Call the model, showing a live spinner (rich) or a plain line so the
    silent 'model is thinking' gap doesn't look like a freeze."""
    if _console is not None:
        # A live spinner; it's cleared automatically before the next print.
        with _console.status(f"[cyan]{label}[/cyan]", spinner="dots"):
            return client.chat(model=model, messages=messages, tools=tools)
    print(f"... {label}", flush=True)
    return client.chat(model=model, messages=messages, tools=tools)


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Normalise an Ollama response message (pydantic OR dict) to a plain dict.

    Also normalises any ``tool_calls`` to ``{"function": {"name", "arguments"}}``
    dicts so they can be safely re-sent to the API and iterated over.
    """
    if isinstance(message, dict):
        data = dict(message)
    elif hasattr(message, "model_dump"):
        data = message.model_dump()
    else:  # very old client fallback
        data = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", "") or "",
        }
        if getattr(message, "tool_calls", None):
            data["tool_calls"] = message.tool_calls

    raw_calls = data.get("tool_calls")
    if raw_calls:
        normalised: list[dict[str, Any]] = []
        for call in raw_calls:
            if isinstance(call, dict):
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args = fn.get("arguments", {})
            else:  # pydantic ToolCall
                fn = call.function
                name = fn.name
                args = fn.arguments
            normalised.append({"function": {"name": name, "arguments": args}})
        data["tool_calls"] = normalised
    return data


def _final_text(message: dict[str, Any]) -> str:
    """Extract the answer text, falling back to a reasoning field if needed.

    Reasoning models (e.g. gpt-oss) sometimes return an empty ``content`` while
    putting text in ``thinking``/``reasoning``; use that rather than show nothing.
    """
    content = (message.get("content") or "").strip()
    if content:
        return content
    for key in ("thinking", "reasoning"):
        alt = (message.get(key) or "").strip()
        if alt:
            return alt
    return ""


def _coerce_args(args: Any) -> dict[str, Any]:
    """Ollama usually returns arguments as a dict; tolerate a JSON string too."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_sources(web_search_json: str) -> list[dict[str, str]]:
    """Pull ``{"title", "url"}`` pairs out of a web_search tool result string."""
    try:
        data = json.loads(web_search_json)
    except (json.JSONDecodeError, TypeError):
        return []
    sources: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("url"):
                sources.append({"title": item.get("title", ""), "url": item["url"]})
    return sources


def _execute_tool(name: str, args: dict[str, Any], config: Config) -> str:
    """Dispatch a tool call to the real local function; return a string result."""
    if name == "web_search":
        query = str(args.get("query", "")).strip()
        if not query:
            return "[error] web_search called without a 'query'."
        num = int(args.get("num_results", config.num_results) or config.num_results)
        results = web_search(
            query,
            num_results=num,
            prefer="searxng",
            searxng_host=config.searxng_host,
            verify=config.request_verify,
        )
        if not results:
            return "[web_search notice] No results (all search backends failed or empty)."
        return json.dumps(results, ensure_ascii=False)

    if name == "fetch_page":
        url = str(args.get("url", "")).strip()
        if not url:
            return "[error] fetch_page called without a 'url'."
        max_chars = int(args.get("max_chars", config.max_page_chars) or config.max_page_chars)
        return fetch_page(url, max_chars=max_chars, verify=config.request_verify)

    return f"[error] Unknown tool: {name}"


def run_agent(
    user_query: str,
    model: str | None = None,
    history: list[dict[str, Any]] | None = None,
    config: Config | None = None,
    max_iterations: int = 6,
    use_tools: bool = True,
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    """Run the full tool-calling loop for one user turn.

    Args:
        user_query: The user's message for this turn.
        model: Model name; defaults to ``config.model``.
        history: Prior message list to continue (for multi-turn chat). If it
            already starts with a system message, it is reused as-is.
        config: Resolved :class:`Config`; loaded from defaults/env if omitted.
        max_iterations: Safety cap on model<->tool round-trips.
        use_tools: If False, no tool schemas are sent (plain chat).

    Returns:
        ``(final_answer, sources, messages)`` where ``sources`` is a list of
        ``{"title", "url"}`` gathered from web_search calls and ``messages`` is
        the full updated conversation (pass it back as ``history`` next turn).
    """
    config = config or load_config()

    # Dispatch to the Groq (OpenAI-compatible) backend if selected.
    if config.backend == "groq":
        return _run_agent_groq(
            user_query,
            model=model,
            history=history,
            config=config,
            max_iterations=max_iterations,
            use_tools=use_tools,
        )

    model = model or config.model

    # Verify HTTPS via the OS trust store so corporate/AV MITM roots (Kaspersky)
    # are trusted without disabling verification.
    if config.use_os_truststore:
        enable_os_truststore()

    # If TLS verification is disabled (e.g. corporate proxy), warn once and mute
    # the per-request urllib3 InsecureRequestWarning noise. Only relevant when
    # tools (which make the web requests) are enabled.
    if use_tools and config.request_verify is False:
        logger.warning(
            "TLS verification is DISABLED (verify_ssl: false). This is insecure; "
            "prefer setting a ca_bundle / REQUESTS_CA_BUNDLE to your proxy's CA."
        )
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001 - best-effort; never fatal
            pass

    client = ollama.Client(host=config.ollama_host)
    tools = build_tool_schemas() if use_tools else None

    messages: list[dict[str, Any]] = list(history) if history else []
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": config.system_prompt or SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_query})

    sources: list[dict[str, str]] = []

    for step in range(1, max_iterations + 1):
        label = (
            "Contacting model (first call may load it on the host)..."
            if step == 1
            else f"Model reasoning over tool results (step {step})..."
        )
        response = _chat_with_status(client, model, messages, tools, label)
        message = _message_to_dict(response["message"])
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            # No tools requested -> this is the final natural-language answer.
            return _final_text(message), sources, messages

        # In --no-search mode we pass tools=None, but some models (e.g. gpt-oss)
        # have BUILT-IN browser tools and emit tool_calls anyway. Never execute
        # them: tell the model tools are off so it answers from its own knowledge.
        if not use_tools:
            for call in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call["function"]["name"],
                        "content": (
                            "[tools disabled] Web tools are unavailable in this "
                            "mode. Answer using your own knowledge; do not call tools."
                        ),
                    }
                )
            continue

        # Execute every tool the model asked for this round.
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = _coerce_args(fn.get("arguments", {}))
            _print_tool_call(step, name, args)

            result = _execute_tool(name, args, config)
            messages.append({"role": "tool", "tool_name": name, "content": result})

            if name == "web_search":
                found = _extract_sources(result)
                sources.extend(found)
                # Optionally auto-fetch the top result for extra detail so the
                # model doesn't need an extra round-trip (config-driven).
                if config.auto_fetch_pages and found:
                    top_url = found[0]["url"]
                    _print_tool_call(step, "fetch_page (auto)", {"url": top_url})
                    page = fetch_page(
                        top_url,
                        max_chars=config.max_page_chars,
                        verify=config.request_verify,
                    )
                    messages.append(
                        {"role": "tool", "tool_name": "fetch_page", "content": page}
                    )

    # Hit the iteration cap: nudge the model to answer NOW using what it has,
    # and make one final tool-free call so it can't request more tools.
    logger.warning("run_agent: hit max_iterations=%d; forcing a final answer.", max_iterations)
    messages.append(
        {
            "role": "user",
            "content": (
                "Stop searching. Using the search snippets and page text above, "
                "state the concrete answer directly now (the actual number/figure), "
                "then cite the source URLs. Do not describe what you tried or call "
                "any more tools."
            ),
        }
    )
    response = _chat_with_status(client, model, messages, None, "Composing final answer...")
    final_message = _message_to_dict(response["message"])
    messages.append(final_message)
    answer = _final_text(final_message)
    if not answer:
        answer = (
            "I couldn't compose a final answer within the tool-call limit, but "
            "here are the sources I found:\n"
            + "\n".join(f"- {s.get('title') or s['url']}: {s['url']}" for s in sources)
        )
    return answer, sources, messages


# ---------------------------------------------------------------------------
# Groq backend (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _groq_http_client(config: Config):
    """An httpx client that trusts the OS store (handles Kaspersky), for Groq.

    httpx doesn't use urllib3, so truststore works here even on Anaconda's old
    urllib3. Falls back to a CA bundle or (last resort) no verification.
    """
    import ssl

    import httpx

    timeout = httpx.Timeout(60.0)
    try:
        import truststore

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return httpx.Client(verify=ctx, timeout=timeout)
    except Exception:  # noqa: BLE001 - truststore optional
        pass
    if config.ca_bundle:
        return httpx.Client(verify=config.ca_bundle, timeout=timeout)
    if config.request_verify is False:
        logger.warning("Groq: TLS verification disabled (insecure).")
        return httpx.Client(verify=False, timeout=timeout)
    return httpx.Client(timeout=timeout)


def _make_groq_client(config: Config):
    """Create an OpenAI SDK client pointed at Groq's API."""
    from openai import OpenAI

    return OpenAI(
        api_key=config.groq_api_key,
        base_url=config.groq_base_url,
        http_client=_groq_http_client(config),
    )


def list_groq_models(config: Config) -> list[str]:
    """Return Groq model ids (empty list if no key or on error)."""
    if not config.groq_api_key:
        return []
    try:
        client = _make_groq_client(config)
        return sorted(m.id for m in client.models.list().data)
    except Exception:  # noqa: BLE001 - best-effort; used for nicer UIs
        return []


def _groq_chat_with_status(client: Any, model: str, messages: list, tools: Any, label: str) -> Any:
    """Groq chat call with a spinner (mirrors _chat_with_status for Ollama)."""
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if _console is not None:
        with _console.status(f"[cyan]{label}[/cyan]", spinner="dots"):
            return client.chat.completions.create(**kwargs)
    print(f"... {label}", flush=True)
    return client.chat.completions.create(**kwargs)


def _run_agent_groq(
    user_query: str,
    model: str | None,
    history: list[dict[str, Any]] | None,
    config: Config,
    max_iterations: int,
    use_tools: bool,
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    """The tool-calling loop against Groq's OpenAI-compatible API.

    Same shape as run_agent(), but uses OpenAI-style messages/tool_calls: the
    assistant returns ``tool_calls`` with ids, and each tool result is a message
    with ``role: "tool"`` and a matching ``tool_call_id``.
    """
    model = model or config.groq_model
    if not config.groq_api_key:
        raise RuntimeError(
            "Groq backend selected but no API key found. Set GROQ_API_KEY in your "
            ".env (or groq_api_key in config.yaml). Get a free key at "
            "https://console.groq.com/keys"
        )

    client = _make_groq_client(config)
    tools = build_tool_schemas() if use_tools else None

    messages: list[dict[str, Any]] = list(history) if history else []
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": config.system_prompt or SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_query})

    sources: list[dict[str, str]] = []

    for step in range(1, max_iterations + 1):
        label = "Contacting Groq..." if step == 1 else f"Groq reasoning (step {step})..."
        response = _groq_chat_with_status(client, model, messages, tools, label)
        msg = response.choices[0].message
        tool_calls = list(msg.tool_calls or [])

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            return (msg.content or "").strip(), sources, messages

        for tc in tool_calls:
            name = tc.function.name
            args = _coerce_args(tc.function.arguments)
            _print_tool_call(step, name, args)
            result = _execute_tool(name, args, config)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            if name == "web_search":
                found = _extract_sources(result)
                sources.extend(found)
                if config.auto_fetch_pages and found:
                    top_url = found[0]["url"]
                    _print_tool_call(step, "fetch_page (auto)", {"url": top_url})
                    page = fetch_page(
                        top_url, max_chars=config.max_page_chars, verify=config.request_verify
                    )
                    # Can't use role:tool without a model tool_call id -> add as context.
                    messages.append(
                        {"role": "user", "content": f"[auto-fetched page for {top_url}]\n{page}"}
                    )

    # Iteration cap: force a final answer.
    logger.warning("run_agent(groq): hit max_iterations=%d; forcing a final answer.", max_iterations)
    messages.append(
        {
            "role": "user",
            "content": (
                "Stop searching. Using the search snippets and page text above, "
                "state the concrete answer directly now (the actual number/figure), "
                "then cite the source URLs. Do not describe what you tried or call "
                "any more tools."
            ),
        }
    )
    response = _groq_chat_with_status(client, model, messages, None, "Composing final answer...")
    final = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final})
    if not final.strip():
        final = "I couldn't compose a final answer, but here are the sources:\n" + "\n".join(
            f"- {s.get('title') or s['url']}: {s['url']}" for s in sources
        )
    return final.strip(), sources, messages
