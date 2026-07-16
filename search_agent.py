"""CLI entrypoint for the local Ollama web-search agent.

Usage:
    python search_agent.py "what is the price of iphone 16 in india"
    python search_agent.py --chat
    python search_agent.py --model qwen2.5 "latest news on ..."
    python search_agent.py --no-search "explain quicksort"   # plain chat, no tools
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import ollama

from agent.config import Config, load_config
from agent.ollama_client import list_models, run_agent

# --- optional pretty output ---------------------------------------------------
try:
    from rich.console import Console
    from rich.markdown import Markdown

    _console: Console | None = Console()
except ImportError:  # pragma: no cover - rich is optional
    _console = None


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so non-ASCII answers don't crash on Windows.

    The default Windows console codec (cp1252) can't encode many characters an
    LLM may produce (emoji, curly quotes, etc.), which raises UnicodeEncodeError.
    Reconfiguring to UTF-8 with error replacement avoids that entirely.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _configure_logging() -> None:
    """Quiet root logging, but surface our own INFO logs (e.g. backend used)."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("tools").setLevel(logging.INFO)
    logging.getLogger("agent").setLevel(logging.INFO)


def _dedupe_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop duplicate sources by URL, preserving first-seen order."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for src in sources:
        url = src.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(src)
    return unique


def _print_answer(answer: str, sources: list[dict[str, str]]) -> None:
    """Render the final answer (as Markdown if rich is available) + sources."""
    text = answer or "*(no answer produced)*"
    if _console is not None:
        _console.rule("[bold green]Answer")
        _console.print(Markdown(text))
    else:
        print("\n=== Answer ===")
        print(text)

    unique = _dedupe_sources(sources)
    if not unique:
        return
    if _console is not None:
        _console.print("\n[bold]Sources:[/bold]")
        for src in unique:
            title = src.get("title") or src["url"]
            _console.print(f"  • {title} — [link={src['url']}]{src['url']}[/link]")
    else:
        print("\nSources:")
        for src in unique:
            title = src.get("title") or src["url"]
            print(f"  - {title} — {src['url']}")


def _print_error(message: str) -> None:
    """Print a clean error (red via rich if available) to stderr."""
    if _console is not None:
        _console.print(f"[bold red]Error:[/bold red] {message}")
    else:
        print(f"Error: {message}", file=sys.stderr)


# Names/substrings that hint a model can do tool-calling (best-effort suggestions).
_TOOL_HINTS = ("llama3.1", "llama3.2", "qwen", "mistral", "gpt-oss", "firefunction", "command-r")


def _print_model_not_found(model: str, config: Config) -> None:
    """Explain a 404 model error and list what's actually on the host."""
    _print_error(f"model '{model}' was not found on the Ollama host ({config.ollama_host}).")
    available = list_models(config.ollama_host)
    if not available:
        print(
            "  Could not list models on the host. Check OLLAMA_HOST/.env and that "
            "Ollama is running and reachable.",
            file=sys.stderr,
        )
        return
    print("\nAvailable models on the host:", file=sys.stderr)
    for name in available:
        print(f"  - {name}", file=sys.stderr)
    likely = [n for n in available if any(h in n.lower() for h in _TOOL_HINTS)]
    if likely:
        print("\nLikely tool-capable (good for web search):", file=sys.stderr)
        for name in likely:
            print(f"  - {name}", file=sys.stderr)
    print(
        f'\nTry:  python search_agent.py --model {(likely or available)[0]} "your question"',
        file=sys.stderr,
    )


def _single_shot(query: str, model: str, config: Config, use_tools: bool) -> None:
    """Answer one question and exit."""
    answer, sources, _ = run_agent(
        query, model=model, config=config, use_tools=use_tools
    )
    _print_answer(answer, sources)


def _chat_mode(model: str, config: Config, use_tools: bool) -> None:
    """Interactive multi-turn chat; conversation history persists across turns."""
    banner = f"Interactive chat with '{model}' (tools {'ON' if use_tools else 'OFF'}). Type 'exit' or 'quit' to stop."
    if _console is not None:
        _console.print(f"[bold]{banner}[/bold]")
    else:
        print(banner)

    history: list[dict[str, Any]] | None = None
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not user_input:
            continue

        answer, sources, history = run_agent(
            user_input,
            model=model,
            config=config,
            history=history,
            use_tools=use_tools,
        )
        _print_answer(answer, sources)


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to single-shot or chat mode."""
    parser = argparse.ArgumentParser(
        description="Local web-search agent for Ollama models (SearxNG + DuckDuckGo fallback).",
    )
    parser.add_argument("query", nargs="?", help="A single question to answer.")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode.")
    parser.add_argument("--model", help="Override the model from config at runtime.")
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web tools and just chat normally (for comparison/testing).",
    )
    parser.add_argument(
        "--backend", choices=["ollama", "groq"], help="LLM backend (overrides config)."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file (default: config.yaml)."
    )
    args = parser.parse_args(argv)

    _force_utf8_output()
    _configure_logging()

    config = load_config(args.config)
    if args.backend:
        config.backend = args.backend
    # Pick the right default model for the active backend.
    default_model = config.groq_model if config.backend == "groq" else config.model
    model = args.model or default_model
    use_tools = not args.no_search

    if not args.chat and not args.query:
        parser.print_help()
        return 1

    try:
        if args.chat:
            _chat_mode(model, config, use_tools)
        else:
            _single_shot(args.query, model, config, use_tools)
    except ollama.ResponseError as exc:
        # e.g. wrong/missing --model name -> 404 "model not found".
        if getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower():
            _print_model_not_found(model, config)
        else:
            _print_error(f"Ollama returned an error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net for a clean CLI message
        name = type(exc).__name__
        if config.backend == "groq":
            _print_error(f"Groq error: {exc}")
            return 1
        if "Connect" in name or "Connection" in name or "Timeout" in name:
            _print_error(
                f"Could not reach the Ollama host at {config.ollama_host}. "
                "Check OLLAMA_HOST/.env and that Ollama is running and reachable."
            )
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
