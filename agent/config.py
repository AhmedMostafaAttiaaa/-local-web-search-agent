"""Configuration loading for the Ollama search agent.

Values are resolved with this precedence (highest wins):

    1. Environment variables (including any loaded from a .env file)
    2. Values in config.yaml
    3. Built-in defaults

Only a few fields have environment-variable overrides (see ``ENV_MAP``); the
rest come from config.yaml or the defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- optional deps: fail soft so the module still imports without them --------
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - only hit when pyyaml is missing
    yaml = None  # type: ignore

try:
    from dotenv import load_dotenv

    # Load a .env from the current working directory (if present) into os.environ.
    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is optional
    pass


# Built-in defaults used when a value is absent from both config.yaml and env.
DEFAULTS: dict[str, Any] = {
    # Which LLM backend to use: "ollama" (local) or "groq" (cloud API).
    "backend": "ollama",
    "ollama_host": "http://localhost:11434",
    "model": "llama3.1",
    # Groq (OpenAI-compatible API). Needs an API key from https://console.groq.com
    # gpt-oss-20b handles tool-calling reliably; some Llama models on Groq don't.
    "groq_model": "openai/gpt-oss-20b",
    "groq_base_url": "https://api.groq.com/openai/v1",
    "groq_api_key": "",
    # Optional: override the built-in system prompt (empty -> use the default).
    "system_prompt": "",
    "searxng_host": "http://localhost:8080",
    "num_results": 5,
    "auto_fetch_pages": False,
    "max_page_chars": 3000,
    # TLS verification for outbound HTTP(S) requests (SearxNG/DDG/page fetch).
    # Behind a corporate TLS-intercepting proxy you can either point `ca_bundle`
    # at the proxy's root CA (secure) or set `verify_ssl: false` (insecure).
    "verify_ssl": True,
    "ca_bundle": "",
    # Verify using the OS trust store (via the `truststore` package). This trusts
    # whatever your OS trusts (incl. corporate/AV MITM roots like Kaspersky), so
    # HTTPS works securely without disabling verification. No-op if unavailable.
    "use_os_truststore": True,
}

# Config field -> environment variable that overrides it.
ENV_MAP: dict[str, str] = {
    "backend": "AGENT_BACKEND",
    "ollama_host": "OLLAMA_HOST",
    "model": "OLLAMA_MODEL",
    "groq_model": "GROQ_MODEL",
    "groq_api_key": "GROQ_API_KEY",
    "searxng_host": "SEARXNG_HOST",
    "ca_bundle": "REQUESTS_CA_BUNDLE",  # standard var `requests` also honours
}


@dataclass
class Config:
    """Resolved runtime configuration for the agent."""

    backend: str
    ollama_host: str
    model: str
    groq_model: str
    groq_base_url: str
    groq_api_key: str
    system_prompt: str
    searxng_host: str
    num_results: int
    auto_fetch_pages: bool
    max_page_chars: int
    verify_ssl: bool
    ca_bundle: str
    use_os_truststore: bool

    @property
    def request_verify(self) -> bool | str:
        """Value to pass to requests' ``verify=`` (CA path, or bool)."""
        if self.ca_bundle:
            return self.ca_bundle
        return self.verify_ssl


def _as_bool(value: Any) -> bool:
    """Coerce common truthy/falsey representations to a real bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: str | os.PathLike[str] = "config.yaml") -> Config:
    """Load configuration from ``path`` (if it exists), env vars, and defaults.

    Args:
        path: Path to a YAML config file. Missing file -> defaults + env only.

    Returns:
        A fully-populated :class:`Config` instance.
    """
    data: dict[str, Any] = dict(DEFAULTS)

    cfg_path = Path(path)
    if cfg_path.is_file():
        if yaml is None:
            print("[config] pyyaml not installed; ignoring config.yaml and using defaults/env.")
        else:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                print(f"[config] {cfg_path} is not a mapping; ignoring it.")
                loaded = {}
            for key, value in loaded.items():
                if key in data and value is not None:
                    data[key] = value

    # Environment variables take priority over the file for mapped fields.
    for field, env_name in ENV_MAP.items():
        env_value = os.environ.get(env_name)
        if env_value:
            data[field] = env_value

    # Normalise types (YAML/env can yield strings).
    data["num_results"] = int(data["num_results"])
    data["max_page_chars"] = int(data["max_page_chars"])
    data["auto_fetch_pages"] = _as_bool(data["auto_fetch_pages"])
    data["verify_ssl"] = _as_bool(data["verify_ssl"])
    data["ca_bundle"] = str(data["ca_bundle"] or "")
    data["use_os_truststore"] = _as_bool(data["use_os_truststore"])
    data["backend"] = str(data["backend"] or "ollama").strip().lower()
    data["groq_api_key"] = str(data["groq_api_key"] or "")
    data["system_prompt"] = str(data["system_prompt"] or "")

    return Config(**data)  # type: ignore[arg-type]


_truststore_injected = False


def enable_os_truststore() -> bool:
    """Route Python's TLS verification through the OS trust store (idempotent).

    Uses the `truststore` package so HTTPS is verified against certificates the
    OS trusts — including corporate/AV interception roots (e.g. Kaspersky) that
    Python's bundled CA list doesn't know about. Returns True if active.

    IMPORTANT: `truststore` is incompatible with urllib3 v1.x (it triggers a
    RecursionError in ssl's verify_mode setter, seen e.g. in Anaconda envs), so
    we only inject when urllib3 >= 2 is present. When skipped, the tools fall
    back to an adaptive verify-then-retry strategy (see tools/_http.py).
    """
    global _truststore_injected
    if _truststore_injected:
        return True
    try:
        import urllib3

        major = int(str(urllib3.__version__).split(".")[0])
        if major < 2:
            return False  # would recurse; leave normal verification in place
    except Exception:  # noqa: BLE001 - if we can't tell, be conservative and skip
        return False
    try:
        import truststore

        truststore.inject_into_ssl()
        _truststore_injected = True
        return True
    except Exception:  # noqa: BLE001 - optional; fall back to normal verification
        return False
