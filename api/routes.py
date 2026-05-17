"""
FastAPI route definitions for the research assistant.

Single endpoint: POST /research
  - Accepts JSON body: {"topic": "<query string>"}
  - Runs the full LangGraph pipeline
  - Returns the final markdown report as a plain string in JSON
"""

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from graph.builder import graph

router = APIRouter()


class ResearchRequest(BaseModel):
    """Request body schema for the /research endpoint."""
    topic: str

    # Reject blank or whitespace-only topics early.
    @field_validator("topic")
    @classmethod
    def topic_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be empty")
        return v.strip()


class ResearchResponse(BaseModel):
    """Response body schema returned by the /research endpoint."""
    topic: str
    report: str


@router.post("/research", response_model=ResearchResponse, summary="Run a research pipeline")
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """
    Trigger the multi-agent research pipeline for the given topic.

    The graph runs synchronously inside the async handler (acceptable for
    this single-worker demo; swap to run_in_executor for production scale).
    """
    # Each invocation gets a unique thread_id so MemorySaver isolates its state.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "topic": request.topic,
        "raw_results": None,
        "summary": None,
        "final_report": None,
        "research_iterations": 0,
    }

    try:
        # invoke() runs the graph to completion and returns the final state.
        final_state = graph.invoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    report = final_state.get("final_report") or "No report generated."
    return ResearchResponse(topic=request.topic, report=report)
