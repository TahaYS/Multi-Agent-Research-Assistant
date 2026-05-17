"""
Report Writer Agent — compiles a structured markdown research report.

Reads the topic, summary, and raw results from state, then asks the LLM to
produce a well-formatted report with three sections: Overview, Key Findings,
and Sources.  The finished report is stored in state as `final_report`.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState


def _get_llm() -> ChatGroq:
    """Instantiate ChatGroq using the GROQ_API_KEY from the environment."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.4,
    )


def report_writer_agent(state: ResearchState) -> dict:
    """LangGraph node: write a structured markdown report from the summary."""

    llm = _get_llm()
    topic = state["topic"]
    summary = state["summary"]
    raw_results = state.get("raw_results", "")

    messages = [
        SystemMessage(
            content=(
                "You are a professional research report writer. "
                "Produce a clean, structured markdown report. "
                "The report MUST contain exactly these three sections:\n"
                "1. ## Overview — a short paragraph introducing the topic.\n"
                "2. ## Key Findings — the bullet-point findings.\n"
                "3. ## Sources — list any URLs or source names found in the raw results; "
                "if none are identifiable, write 'Sources derived from DuckDuckGo search'."
            )
        ),
        HumanMessage(
            content=(
                f"# Research Report: {topic}\n\n"
                f"Key findings summary:\n{summary}\n\n"
                f"Raw search text (for source extraction):\n{raw_results[:2000]}"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {"final_report": response.content}
