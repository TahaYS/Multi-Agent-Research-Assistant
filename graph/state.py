"""
Shared state schema for the research graph.

Every agent node reads from and writes to this TypedDict.  LangGraph merges
return dicts with the existing state at each step — fields not returned by a
node are left unchanged.
"""

from typing import Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # The user-supplied research topic.
    topic: str

    # Raw text returned by Wikipedia query.
    raw_results: Optional[str]

    # Bullet-point summary produced by the Summarizer Agent.
    summary: Optional[str]

    # Final markdown report produced by the Report Writer Agent.
    final_report: Optional[str]

    # How many times the Research Agent has run; used by the router to
    # decide whether to loop back for another search pass.
    research_iterations: int
