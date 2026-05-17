"""
Streamlit frontend for the Multi-Agent Research Assistant.

Calls the LangGraph pipeline directly — no FastAPI server required.
Works both locally and on Streamlit Cloud (GROQ_API_KEY supplied via
st.secrets on Cloud, or via the environment for local dev).

Run locally with:  streamlit run streamlit_app.py
"""

import os
import uuid
import streamlit as st


def _inject_api_key() -> None:
    """
    Ensure GROQ_API_KEY is in os.environ before the graph is invoked.

    On Streamlit Cloud, secrets set in the dashboard are available via
    st.secrets but are NOT always auto-injected into os.environ, so we
    copy the key explicitly.  For local dev the key is expected to already
    be in the environment (set manually or via a .env-aware shell).
    """
    if "GROQ_API_KEY" not in os.environ:
        try:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error(
                "**GROQ_API_KEY is not set.** "
                "Add it to Streamlit Cloud → Settings → Secrets, "
                "or export it in your local shell before running."
            )
            st.stop()


def main() -> None:
    """Entry point called by streamlit_app.py."""

    # set_page_config must be the first Streamlit call on every script run.
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🔬",
        layout="centered",
    )

    # Inject the API key before any graph work; may call st.stop() on failure.
    _inject_api_key()

    # Import is cached by Python after the first run — safe to call each rerender.
    from graph.builder import graph

    st.title("🔬 Multi-Agent Research Assistant")
    st.caption("Powered by LangGraph · ChatGroq (Llama 3.1) · Wikipedia")

    # ── Input form ────────────────────────────────────────────────────────────

    topic = st.text_input(
        label="Research topic",
        placeholder="e.g. Quantum computing breakthroughs in 2024",
        help="Enter any topic you want researched. The pipeline will search, summarize, and write a report.",
    )

    run_button = st.button("Run Research", type="primary", disabled=not topic.strip())

    # ── Pipeline execution ────────────────────────────────────────────────────

    if run_button and topic.strip():
        with st.spinner("Running multi-agent research pipeline… this may take 15–30 seconds."):
            # Unique thread_id isolates this run's MemorySaver checkpoint from others.
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            initial_state = {
                "topic": topic.strip(),
                "raw_results": None,
                "summary": None,
                "final_report": None,
                "research_iterations": 0,
            }

            try:
                final_state = graph.invoke(initial_state, config=config)
                report = final_state.get("final_report") or "No report was generated."

                st.success("Research complete!")
                st.markdown("---")
                st.markdown(report)

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
