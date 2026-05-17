"""
Streamlit frontend for the Multi-Agent Research Assistant.

Renders a text input for the research topic, a submit button, and displays
the markdown report returned by the FastAPI backend once research completes.

Run with:  streamlit run frontend/app.py
The FastAPI server must be running at http://localhost:8000.
"""

import requests
import streamlit as st

# URL of the FastAPI research endpoint.
API_URL = "http://localhost:8000/research"

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Multi-Agent Research Assistant")
st.caption(
    "Powered by LangGraph · ChatGroq (Llama 3.1) · Wikipedia"
)

# ── Input form ────────────────────────────────────────────────────────────────

topic = st.text_input(
    label="Research topic",
    placeholder="e.g. Quantum computing breakthroughs in 2024",
    help="Enter any topic you want researched. The pipeline will search, summarize, and write a report.",
)

run_button = st.button("Run Research", type="primary", disabled=not topic.strip())

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_button and topic.strip():
    with st.spinner("Running multi-agent research pipeline… this may take 15–30 seconds."):
        try:
            response = requests.post(
                API_URL,
                json={"topic": topic.strip()},
                timeout=120,  # give the pipeline enough time to complete
            )
            response.raise_for_status()
            data = response.json()

            st.success("Research complete!")

            # Render the markdown report returned by the Report Writer Agent.
            st.markdown("---")
            st.markdown(data["report"])

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI server at `http://localhost:8000`. "
                "Please start it with: `uvicorn api.app:app --reload`"
            )
        except requests.exceptions.Timeout:
            st.error("The request timed out. The pipeline may be overloaded — try again.")
        except requests.exceptions.HTTPError as exc:
            detail = exc.response.json().get("detail", str(exc))
            st.error(f"API error: {detail}")
