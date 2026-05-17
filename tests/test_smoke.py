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


# ── DuckDuckGo retry logic ────────────────────────────────────────────────────

def test_retry_succeeds_after_rate_limit(monkeypatch):
    """Search should succeed on the third attempt after two rate-limit errors."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    call_count = {"n": 0}

    def fake_run(topic):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("DuckDuckGo 202 Ratelimit")
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
    mock_search.run.side_effect = Exception("202 Ratelimit")
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        ra._search_with_retry("test topic")


def test_non_rate_limit_error_not_retried(monkeypatch):
    """Non-rate-limit exceptions should propagate immediately without retrying."""
    import agents.research_agent as ra
    from unittest.mock import MagicMock

    call_count = {"n": 0}

    def fake_run(topic):
        call_count["n"] += 1
        raise ValueError("some other error")

    mock_search = MagicMock()
    mock_search.run.side_effect = fake_run
    monkeypatch.setattr(ra, "_search", mock_search)
    monkeypatch.setattr(ra, "_RETRY_DELAY", 0)

    with pytest.raises(ValueError, match="some other error"):
        ra._search_with_retry("test topic")

    assert call_count["n"] == 1  # must not have retried


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
