# Ollama Web-Search Agent

A **100% free, local** tool-calling agent that lets any tool-capable Ollama
model **search the live internet** instead of relying only on its training data.

-  **Primary search backend:** [SearxNG](https://github.com/searxng/searxng) (self-hosted, open-source metasearch) via Docker — **no API keys**.
-  **Fallback:** DuckDuckGo HTML scrape (no API key) if SearxNG is unreachable.
-  **Page-fetch tool:** downloads a URL and extracts clean readable text so the model can read exact details (like a price) beyond the snippet.
-  **Native Ollama tool-calling:** the model itself decides when to call `web_search` and/or `fetch_page`, across multiple turns.
-  **CLI + interactive chat** modes, configurable via `config.yaml` / `.env`.

```
ollama-search-agent/
├── README.md
├── requirements.txt
├── docker-compose.yml          # SearxNG service (JSON enabled via mounted config)
├── searxng/
│   └── settings.yml            # enables formats: [html, json] + disables limiter
├── config.example.yaml         # copy to config.yaml
├── config.yaml                 # active config
├── search_agent.py             # CLI entrypoint (single query + --chat)
├── tools/
│   ├── __init__.py
│   ├── web_search.py           # SearxNG search + DuckDuckGo fallback
│   └── fetch_page.py           # fetch & clean page text
├── agent/
│   ├── __init__.py
│   ├── ollama_client.py        # tool schemas + the tool-calling loop
│   └── config.py               # loads config.yaml / .env
└── tests/
    └── test_tools.py
```

---

## 1. Prerequisites

- **Python 3.10+**
- **Docker** + Docker Compose (for SearxNG)
- **Ollama** installed and running, with a **tool-calling-capable model** pulled, e.g.:
  ```bash
  ollama pull llama3.1
  # other good tool-callers: qwen2.5, mistral-nemo, llama3.2, firefunction-v2
  ```
  > Not every model supports tools. If you get errors about tools, try `llama3.1` / `qwen2.5`.

Create a `.env` (or edit `config.yaml`) with your Ollama host, e.g.
`OLLAMA_HOST=http://localhost:11434`. Point it at a remote host if your Ollama
runs elsewhere on your network.

---

## 2. Setup (step by step)

```bash
# 0. (recommended) create + activate an isolated environment. Use EITHER venv OR
#    conda — but a DEDICATED one, not conda `base` (its old urllib3 v1 breaks the
#    secure TLS path). A fresh env pulls urllib3>=2 -> secure verification works.

# --- Option A: conda ---
conda env create -f environment.yml   # creates env "websearch" + installs deps
conda activate websearch
# (deps are installed by environment.yml, so you can skip step 1 below)

# --- Option B: venv ---
python -m venv .venv
.venv\Scripts\activate            # Windows CMD  (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/Scripts/activate   # Git Bash
# source .venv/bin/activate       # macOS/Linux

# 1. Install Python dependencies (venv only; conda did this already)
pip install -r requirements.txt

# 2. Start SearxNG (JSON API is enabled automatically via searxng/settings.yml)
docker compose up -d

# 3. (recommended) set a real secret key for SearxNG
#    edit searxng/settings.yml -> server.secret_key: "<some-random-string>"
#    then: docker compose restart

# 4. Create your config
cp config.example.yaml config.yaml     # (already present in this repo)
#    edit config.yaml -> set `model:` to a model you've actually pulled

# 5. Verify Ollama connectivity + see available models
python test_ollama.py                  # prints models, writes models.md
```

### Enabling the SearxNG JSON API (already handled here)

SearxNG **disables the JSON output format by default** — a plain
`GET /search?format=json` returns HTTP `403`. This project fixes that for you by
mounting [`searxng/settings.yml`](searxng/settings.yml), which sets:

```yaml
use_default_settings: true
server:
  limiter: false          # allow local programmatic requests
search:
  formats:
    - html
    - json                # <-- enables the JSON API this agent uses
```

Confirm it works:

```bash
curl "http://localhost:8080/search?q=test&format=json"
# should return JSON with a "results" array (not a 403 / HTML page)
```

If you change `searxng/settings.yml`, run `docker compose restart` to apply it.

---

## 3. Usage

**Single question (one-shot):**
```bash
python search_agent.py "what is the price of iphone 16 in india"
```

**Interactive chat (keeps history across turns):**
```bash
python search_agent.py --chat
# type 'exit' or 'quit' to leave
```

**Override the model at runtime:**
```bash
python search_agent.py --model qwen2.5 "latest news about the mars rover"
```

**Plain chat, no web tools (for comparison/testing):**
```bash
python search_agent.py --no-search "explain the quicksort algorithm"
```

### Using Groq instead of Ollama (cloud, no local model needed)

The agent also supports [Groq](https://console.groq.com) — a fast, OpenAI-compatible
API with free open models (Llama, Qwen, gpt-oss) that support tool-calling.

1. Get a free key at https://console.groq.com/keys and add it to `.env`:
   ```bash
   GROQ_API_KEY=gsk_...
   ```
2. Run with the Groq backend:
   ```bash
   python search_agent.py --backend groq "usd to egp rate today"
   python search_agent.py --backend groq --model openai/gpt-oss-20b "latest F1 winner"
   ```

Or set it permanently in `config.yaml`: `backend: "groq"`. Everything else (web
search, sources, TLS handling) works identically.

**Model note:** prefer `openai/gpt-oss-20b` (the default) or `qwen/qwen3-32b` —
they do tool-calling reliably on Groq. Some Llama models (e.g.
`llama-3.3-70b-versatile`) intermittently emit malformed tool calls and fail with
`tool_use_failed`.

### Sample output (abridged)

```
🛠  Tool call · step 1
┌──────────────────────────────────────────────┐
│ web_search({"query": "iphone 16 price india"})│
└──────────────────────────────────────────────┘
web_search: used searxng (5 results)

🛠  Tool call · step 2
fetch_page({"url": "https://www.apple.com/in/shop/buy-iphone/iphone-16"})

──────────────────────────── Answer ─────────────────────────────
The iPhone 16 (128 GB) starts at ₹79,900 in India, per Apple India's
official store. The 256 GB is ₹89,900 and the 512 GB is ₹1,09,900.

Sources:
  • Buy iPhone 16 - Apple (IN) — https://www.apple.com/in/shop/buy-iphone/iphone-16
  • ...
```

---

## 4. Configuration

`config.yaml` (see [`config.example.yaml`](config.example.yaml)):

| Field              | Default                   | Meaning                                            |
|--------------------|---------------------------|----------------------------------------------------|
| `ollama_host`      | `http://localhost:11434`  | Ollama server URL                                  |
| `model`            | `llama3.1`                | Model name (must support tool calls)               |
| `searxng_host`     | `http://localhost:8080`   | SearxNG instance URL                               |
| `num_results`      | `5`                       | Search results fetched per query                   |
| `auto_fetch_pages` | `false`                   | Auto-fetch the top result's page after each search |
| `max_page_chars`   | `3000`                    | Max characters returned by `fetch_page`            |
| `verify_ssl`       | `true`                    | Verify TLS certs for web requests                  |
| `ca_bundle`        | `""`                      | Optional path to a CA bundle (corporate proxy)     |

**Env var overrides** (env wins over `config.yaml`): `OLLAMA_HOST`,
`OLLAMA_MODEL`, `SEARXNG_HOST`, `REQUESTS_CA_BUNDLE` (→ `ca_bundle`). These can
live in a `.env` file.

---

## 5. Tests

```bash
pytest -v
# or a quick manual smoke test:
python tests/test_tools.py
```

Network-dependent tests **skip automatically** if SearxNG/DuckDuckGo are
unreachable, so the suite won't hard-fail offline.

---

## 6. Troubleshooting

**SearxNG returns 403 / HTML instead of JSON**
JSON format isn't enabled. Ensure `searxng/settings.yml` has `formats: [html, json]`
and is mounted (`docker compose config` should show the volume), then
`docker compose restart`. Test with the `curl` command in section 2.

**`Connection refused` to SearxNG**
Is the container up? `docker compose ps`. Start it with `docker compose up -d`.
The agent will fall back to DuckDuckGo, but fixing SearxNG gives better results.

**DuckDuckGo returns nothing / rate-limits you**
The HTML endpoint occasionally throttles automated requests. Wait a bit, reduce
`num_results`, and prefer running SearxNG as the primary backend.

**Ollama: model doesn't support tools / no `tool_calls`**
Some models can't do function-calling. Use a tool-capable one
(`ollama pull llama3.1` or `qwen2.5`) and set it in `config.yaml`. If a model
ignores tools, it will just answer from training data — use `--model` to switch.

**`RecursionError: maximum recursion depth exceeded` in `ssl.py`**
This happens when `truststore` runs on **urllib3 v1.x** (e.g. the Anaconda `base`
env ships urllib3 1.26). The tool now detects this and skips `truststore` when
urllib3 < 2, then adaptively retries requests without verification if a local
HTTPS interceptor (Kaspersky) blocks them — so it keeps working. For *secure*
verification via the OS trust store, run in an env with `urllib3>=2`
(`pip install -U urllib3`, or use the standalone Python rather than conda `base`).

**`gpt-oss` keeps calling tools even with `--no-search`**
`gpt-oss` ships with *built-in* browser tools (`web_search`/`fetch_page`) in its
training, so it may emit tool calls even when we send no tool schemas. The agent
handles this: in `--no-search` mode it refuses those calls and tells the model to
answer from its own knowledge, so no web request is actually made. In normal mode
`gpt-oss` conforms to this project's tool schemas (`query`/`url`/…) as expected.

**`Connection refused` to Ollama**
Check `OLLAMA_HOST` (in `.env`/`config.yaml`) and that `ollama serve` is running
and reachable. `python test_ollama.py` lists the models it can see.

**`SSL: CERTIFICATE_VERIFY_FAILED` / `self signed certificate in certificate chain`**
Your network has a TLS-intercepting proxy (common on corporate networks) whose
root CA isn't trusted by Python's default bundle. Two options:

- **Secure (recommended):** export your proxy's root CA and point at it:
  ```bash
  # config.yaml
  ca_bundle: "C:/path/to/corporate-root-ca.pem"
  # or via env (requests honours this natively):
  set REQUESTS_CA_BUNDLE=C:\path\to\corporate-root-ca.pem   # Windows cmd
  ```
  (On Windows you can export the CA from Certificate Manager, or via
  `pip install python-certifi-win32` / your IT-provided `.pem`.)
- **Insecure (quick unblock, dev only):** disable verification:
  ```yaml
  # config.yaml
  verify_ssl: false
  ```
  The agent logs a one-time warning and mutes urllib3's per-request warnings.

Running SearxNG locally over `http://localhost:8080` also sidesteps TLS for the
*search* step (only page fetches then hit HTTPS sites).

**Nothing prints in color / no panels**
`rich` isn't installed. It's optional — the app degrades to plain text.
`pip install rich` to get the nicer output.

---

## 7. Extending it

The tools are decoupled from the agent loop, so adding a tool is 3 steps:

1. Write a plain Python function (e.g. `tools/calculator.py: def calculate(expr)`).
2. Add its JSON schema to `build_tool_schemas()` in
   [`agent/ollama_client.py`](agent/ollama_client.py).
3. Add a branch for it in `_execute_tool(...)`.

Ideas: a calculator, a news-specific search (`web_search` with a `news` engine
filter), a Wikipedia tool, a local-file search, or a currency converter.

**Swapping the search backend:** implement a new `search_*()` in
[`tools/web_search.py`](tools/web_search.py) and wire it into `web_search()` —
the agent loop never needs to change.

---

## How the tool-calling loop works (conceptually)

We hand the model the user's question **plus JSON schemas** describing each tool
(`web_search`, `fetch_page`). The model replies with **either** a normal text
answer **or** a structured `tool_calls` request naming a tool and its arguments.
When it requests a tool, *our code* — not the model — runs the real Python
function locally and captures the output. We append the model's own message and
a `role: "tool"` message carrying that output back into the conversation, then
call the model again. Now the model can *see* the fresh web data and either ask
for another tool (e.g. `fetch_page` on a promising URL) or write its final
answer. We repeat this request→execute→feed-back cycle until the model stops
asking for tools, capping iterations so it can't loop forever. This is why the
model can answer real-time questions it was never trained on: the loop turns the
LLM into a planner that drives real tools, and the tool results become new
context it reasons over before answering.
