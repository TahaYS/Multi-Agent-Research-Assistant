"""
LangGraph graph builder — wires the three agents into a stateful workflow.

Graph topology:
  research_agent
       |
       v
  [router: sufficient?]
       |                \
       | yes             | no (up to MAX_ITERATIONS)
       v                 v
  summarizer_agent   research_agent  (loop back)
       |
       v
  report_writer_agent
       |
       v
      END

Persistent memory across steps is provided by MemorySaver, which checkpoints
the state after every node so the graph can resume from any point.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import ResearchState
from agents.research_agent import research_agent
from agents.summarizer_agent import summarizer_agent
from agents.report_writer_agent import report_writer_agent

# Maximum research loops before forcing progression to the summarizer.
MAX_ITERATIONS = 2
# Minimum character count to consider raw results "sufficient".
MIN_RESULT_LENGTH = 200


def _route_after_research(state: ResearchState) -> str:
    """
    Conditional edge: decide whether research results are good enough.

    Returns 'summarize' if the raw results meet the quality threshold or
    if we have already retried the maximum number of times.
    Returns 'research' to loop back for another search pass otherwise.
    """
    raw = state.get("raw_results") or ""
    iterations = state.get("research_iterations", 0)

    # Force progression after MAX_ITERATIONS to avoid infinite loops.
    if iterations >= MAX_ITERATIONS:
        return "summarize"

    # Results are sufficient when they contain enough text.
    if len(raw) >= MIN_RESULT_LENGTH:
        return "summarize"

    # Loop back for another research attempt.
    return "research"


def build_graph() -> StateGraph:
    """Construct and compile the research assistant graph with memory."""

    # Initialise the graph with our shared state schema.
    workflow = StateGraph(ResearchState)

    # Register each agent as a named node.
    workflow.add_node("research_agent", research_agent)      # web search
    workflow.add_node("summarizer_agent", summarizer_agent)  # distil findings
    workflow.add_node("report_writer_agent", report_writer_agent)  # write report

    # Entry point — always start with the Research Agent.
    workflow.set_entry_point("research_agent")

    # Conditional edge after research: loop back or proceed to summarizer.
    workflow.add_conditional_edges(
        "research_agent",
        _route_after_research,
        {
            "research": "research_agent",   # loop: results insufficient
            "summarize": "summarizer_agent", # proceed: results good enough
        },
    )

    # Linear edge: summarizer feeds into the report writer.
    workflow.add_edge("summarizer_agent", "report_writer_agent")

    # Terminal edge: report writer marks the end of the workflow.
    workflow.add_edge("report_writer_agent", END)

    # Attach MemorySaver to persist state across steps (supports resumption).
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Module-level compiled graph — imported by the API and tests.
graph = build_graph()
