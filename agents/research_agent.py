"""
Research Agent — searches the web using DuckDuckGo and returns raw results.

Receives the research topic from the graph state, runs the search with
automatic retry on rate-limit errors, and stores raw result snippets back
into state for the summarizer to consume.
"""

import time
from langchain_community.tools import DuckDuckGoSearchRun
from graph.state import ResearchState


# One shared search tool instance; DuckDuckGo requires no API key.
_search = DuckDuckGoSearchRun()

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between attempts after a rate-limit error


def _search_with_retry(topic: str) -> str:
    """Run a DuckDuckGo search, retrying up to _MAX_RETRIES times on rate limits."""
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _search.run(topic)
        except Exception as exc:
            # DuckDuckGo rate-limit responses surface as exceptions containing "202"
            # or "Ratelimit" in the message.
            if "202" in str(exc) or "ratelimit" in str(exc).lower():
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
            else:
                raise  # non-rate-limit errors propagate immediately

    raise RuntimeError(
        f"DuckDuckGo search failed after {_MAX_RETRIES} attempts: {last_error}"
    )


def research_agent(state: ResearchState) -> dict:
    """LangGraph node: perform web research on the given topic."""

    topic = state["topic"]

    # Run the DuckDuckGo search with retry logic for rate-limit errors.
    raw_results = _search_with_retry(topic)

    # Increment the iteration counter so the router can detect insufficient results.
    iterations = state.get("research_iterations", 0) + 1

    return {
        "raw_results": raw_results,
        "research_iterations": iterations,
    }
