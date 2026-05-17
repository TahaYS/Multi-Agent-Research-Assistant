"""
Research Agent — queries Wikipedia and returns raw article content.

Receives the research topic from the graph state, runs the query with
automatic retry on transient errors, and stores raw results back into
state for the summarizer to consume.
"""

import time
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from graph.state import ResearchState


# Wikipedia tool requires no API key; top_k_results limits pages fetched.
_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between attempts after a transient error


def _search_with_retry(topic: str) -> str:
    """Run a Wikipedia query, retrying up to _MAX_RETRIES times on transient errors."""
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _search.run(topic)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    raise RuntimeError(
        f"Wikipedia search failed after {_MAX_RETRIES} attempts: {last_error}"
    )


def research_agent(state: ResearchState) -> dict:
    """LangGraph node: perform Wikipedia research on the given topic."""

    topic = state["topic"]

    # Run the Wikipedia query with retry logic for transient errors.
    raw_results = _search_with_retry(topic)

    # Increment the iteration counter so the router can detect insufficient results.
    iterations = state.get("research_iterations", 0) + 1

    return {
        "raw_results": raw_results,
        "research_iterations": iterations,
    }
