"""
Smoke tests — verify core imports, state schema, graph topology, and the API
app can be imported without errors.  These tests do NOT make real network calls
or require a GROQ_API_KEY.
"""

import os
import pytest


# ── State schema ──────────────────────────────────────────────────────────────

def test_state_schema_imports():
    from graph.state import ResearchState
    # Instantiate a minimal valid state to confirm the TypedDict is intact.
    state: ResearchState = {
        "topic": "test topic",
        "raw_results": None,
        "summary": None,
        "final_report": None,
        "research_iterations": 0,
    }
    assert state["topic"] == "test topic"
    assert state["research_iterations"] == 0


# ── Graph builder ─────────────────────────────────────────────────────────────

def test_graph_builds_without_error():
    """The compiled graph object should be importable without a live API key."""
    # Set a dummy key so the import of ChatGroq doesn't raise on missing env var.
    os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-import-test")
    from graph.builder import build_graph
    g = build_graph()
    # A compiled LangGraph exposes .nodes — confirm our three agents are present.
    assert "research_agent" in g.nodes
    assert "summarizer_agent" in g.nodes
    assert "report_writer_agent" in g.nodes


# ── FastAPI app ───────────────────────────────────────────────────────────────

def test_fastapi_app_imports():
    os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-import-test")
    from api.app import app
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


def test_research_route_registered():
    os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-import-test")
    from api.app import app
    routes = [r.path for r in app.routes]
    assert "/research" in routes


# ── Wikipedia retry logic ─────────────────────────────────────────────────────

def test_retry_succeeds_after_transient_error(monkeypatch):
    """Search should succeed on the third attempt after two transient errors."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    call_count = {"n": 0}

    def fake_run(topic):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("Wikipedia network error")
        return "good results"

    mock_search = MagicMock()
    mock_search.run.side_effect = fake_run
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)  # no sleeping in tests

    result = ra._search_with_retry("test topic")
    assert result == "good results"
    assert call_count["n"] == 3


def test_retry_raises_after_max_attempts(monkeypatch):
    """Should raise RuntimeError once all retries are exhausted."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    mock_search = MagicMock()
    mock_search.run.side_effect = Exception("Wikipedia network error")
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        ra._search_with_retry("test topic")


def test_retry_exhausts_all_attempts(monkeypatch):
    """Every attempt should be used before raising — no early bail-out."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    call_count = {"n": 0}

    def fake_run(topic):
        call_count["n"] += 1
        raise Exception("transient error")

    mock_search = MagicMock()
    mock_search.run.side_effect = fake_run
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    with pytest.raises(RuntimeError):
        ra._search_with_retry("test topic")

    assert call_count["n"] == ra._MAX_RETRIES


# ── No-results / parsing-error handling ──────────────────────────────────────

def test_no_results_marker_raises_immediately(monkeypatch):
    """LangChain's no-results sentinel must raise _WikipediaNoResultsError without retrying."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    mock_search = MagicMock()
    mock_search.run.return_value = ra._WIKI_NO_RESULTS_MARKER
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    with pytest.raises(ra._WikipediaNoResultsError):
        ra._search_with_retry("nonexistent topic xkcd123")

    assert mock_search.run.call_count == 1  # must not have retried


def test_empty_result_raises_immediately(monkeypatch):
    """A blank result string is treated the same as no results — no retry."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    mock_search = MagicMock()
    mock_search.run.return_value = "   "
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    with pytest.raises(ra._WikipediaNoResultsError):
        ra._search_with_retry("blank topic")

    assert mock_search.run.call_count == 1


def test_research_agent_sets_final_report_on_no_results(monkeypatch):
    """research_agent must set final_report to the friendly message and not crash."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    mock_search = MagicMock()
    mock_search.run.return_value = ra._WIKI_NO_RESULTS_MARKER
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    state = {"topic": "xkcd nonexistent", "research_iterations": 0}
    result = ra.research_agent(state)

    assert result["final_report"] == ra.FRIENDLY_NO_RESULTS
    assert result["raw_results"] == ""
    assert mock_search.run.call_count == 1  # no retries


def test_router_short_circuits_to_end_when_final_report_set():
    """Router must return 'end' when final_report is already populated."""
    from graph.builder import _route_after_research

    state = {
        "topic": "test",
        "raw_results": "",
        "summary": None,
        "final_report": "No results found for this topic. Please try a more specific or well-known subject.",
        "research_iterations": 1,
    }
    assert _route_after_research(state) == "end"


# ── Request / Response models ─────────────────────────────────────────────────

def test_research_request_rejects_empty_topic():
    from pydantic import ValidationError
    from api.routes import ResearchRequest
    with pytest.raises(ValidationError):
        ResearchRequest(topic="   ")


def test_research_request_strips_whitespace():
    from api.routes import ResearchRequest
    req = ResearchRequest(topic="  quantum computing  ")
    assert req.topic == "quantum computing"
