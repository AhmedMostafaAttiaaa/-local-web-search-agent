"""A very basic Streamlit UI for the local web-search agent.

Run it with:
    streamlit run streamlit_app.py

It reuses the exact same pieces as the CLI (`load_config`, `run_agent`), so the
model, search backends, and TLS handling all behave identically.
"""
from __future__ import annotations

import streamlit as st

from agent.config import enable_os_truststore, load_config
from agent.ollama_client import list_models, run_agent


@st.cache_resource
def _startup():
    """Load config once and enable secure TLS (OS trust store)."""
    cfg = load_config()
    enable_os_truststore()
    return cfg


def _dedupe(sources: list[dict]) -> list[dict]:
    """Drop duplicate sources by URL, preserving order."""
    seen: set[str] = set()
    out: list[dict] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append(s)
    return out


cfg = _startup()

st.set_page_config(page_title="Local Web-Search Agent", page_icon="🔎")
st.title("🔎 Local Web-Search Agent")
st.caption("Ask anything — the model searches the live web (Ollama + SearxNG / DuckDuckGo).")

# --- Sidebar: settings -------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    models = list_models(cfg.ollama_host)
    if models:
        default_index = models.index(cfg.model) if cfg.model in models else 0
        model = st.selectbox("Model", models, index=default_index)
    else:
        st.warning(f"Couldn't reach Ollama at {cfg.ollama_host}.")
        model = st.text_input("Model", value=cfg.model)

    use_tools = st.checkbox("Enable web search", value=True)

    st.divider()
    st.caption(f"**Ollama:** {cfg.ollama_host}")
    st.caption(f"**SearxNG:** {cfg.searxng_host}")

# --- Main: ask a question ----------------------------------------------------
question = st.text_input(
    "Your question",
    placeholder="what is the usd to egp rate today?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking… (searching the web if needed)"):
        try:
            answer, sources, messages = run_agent(
                question, model=model, config=cfg, use_tools=use_tools
            )
        except Exception as exc:  # noqa: BLE001 - show any error in the UI
            st.error(f"Error: {exc}")
        else:
            st.markdown("### Answer")
            st.markdown(answer or "_(no answer produced)_")

            unique = _dedupe(sources)
            if unique:
                st.markdown("### Sources")
                for s in unique:
                    title = s.get("title") or s["url"]
                    st.markdown(f"- [{title}]({s['url']})")

            # Optional peek at what the agent did (which tools it called).
            tool_calls = [
                (m.get("tool_calls") or [])
                for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            flat = [tc for calls in tool_calls for tc in calls]
            if flat:
                with st.expander(f"🔧 Agent steps ({len(flat)} tool call(s))"):
                    for i, tc in enumerate(flat, 1):
                        fn = tc.get("function", {})
                        st.write(f"{i}. **{fn.get('name')}** — {fn.get('arguments')}")
elif not question.strip():
    st.info("Type a question above and press **Ask**.")
