"""
Research Agent — queries Wikipedia and returns raw article content.

Receives the research topic from the graph state, runs the query with
automatic retry on transient errors, and stores raw results back into
state for the summarizer to consume.

Deterministic failures (no matching article, parsing errors) are caught
before the retry loop and written directly to `final_report` so the graph
can route straight to END without running the summarizer or report writer.
"""

import time
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from graph.state import ResearchState


# Wikipedia tool requires no API key; top_k_results limits pages fetched.
_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between attempts after a transient error

# Exact string LangChain's WikipediaAPIWrapper returns when no pages match.
_WIKI_NO_RESULTS_MARKER = "No good Wikipedia Search Result was found"

# User-facing message written to final_report on a no-results outcome.
FRIENDLY_NO_RESULTS = (
    "No results found for this topic. "
    "Please try a more specific or well-known subject."
)


class _WikipediaNoResultsError(Exception):
    """Raised when Wikipedia returns no usable content — must not be retried."""


def _search_with_retry(topic: str) -> str:
    """
    Run a Wikipedia query, retrying up to _MAX_RETRIES times on transient errors.

    Raises _WikipediaNoResultsError immediately (without retrying) when the
    wrapper signals that no article matched the topic.
    """
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = _search.run(topic)

            # The wrapper returns this sentinel string instead of raising when
            # PageError or DisambiguationError occurs internally — treat it as
            # a deterministic failure and skip the retry loop.
            if not result.strip() or _WIKI_NO_RESULTS_MARKER in result:
                raise _WikipediaNoResultsError(result)

            return result

        except _WikipediaNoResultsError:
            raise  # deterministic — propagate immediately, do not retry

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
    iterations = state.get("research_iterations", 0) + 1

    try:
        raw_results = _search_with_retry(topic)
    except _WikipediaNoResultsError:
        # Set final_report directly so the graph router can send the run to END,
        # bypassing the summarizer and report writer entirely.
        return {
            "raw_results": "",
            "final_report": FRIENDLY_NO_RESULTS,
            "research_iterations": iterations,
        }

    # Increment the iteration counter so the router can detect insufficient results.
    return {
        "raw_results": raw_results,
        "research_iterations": iterations,
    }
