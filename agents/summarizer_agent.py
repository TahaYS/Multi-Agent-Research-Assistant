"""
Summarizer Agent — distills raw search results into concise key findings.

Takes the raw Wikipedia text from state, sends it to the LLM with a focused
prompt, and writes a bullet-point summary back into state for the report writer.
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
        temperature=0.3,
    )


def summarizer_agent(state: ResearchState) -> dict:
    """LangGraph node: summarize raw research results into key findings."""

    llm = _get_llm()
    topic = state["topic"]
    raw_results = state["raw_results"]

    messages = [
        SystemMessage(
            content=(
                "You are a concise research summarizer. "
                "Extract the most important facts and insights from the provided text. "
                "Return a bullet-point list of key findings — no preamble, no filler."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Raw research results:\n{raw_results}\n\n"
                "Summarize the key findings as a bullet-point list."
            )
        ),
    ]

    response = llm.invoke(messages)

    return {"summary": response.content}
