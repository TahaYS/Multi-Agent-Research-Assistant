"""
Research Agent — searches the web using DuckDuckGo and returns raw results.

Receives the research topic from the graph state, runs up to 5 searches,
and stores raw result snippets back into state for the summarizer to consume.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from graph.state import ResearchState


# One shared search tool instance; DuckDuckGo requires no API key.
_search = DuckDuckGoSearchRun()


def research_agent(state: ResearchState) -> dict:
    """LangGraph node: perform web research on the given topic."""

    topic = state["topic"]

    # Run the DuckDuckGo search — returns a single string of concatenated results.
    raw_results = _search.run(topic)

    # Increment the iteration counter so the router can detect insufficient results.
    iterations = state.get("research_iterations", 0) + 1

    return {
        "raw_results": raw_results,
        "research_iterations": iterations,
    }
