# Multi-Agent Research Assistant

A production-grade research pipeline built with **LangGraph**, **LangChain**, **ChatGroq (Llama 3.1)**, **Wikipedia**, **FastAPI**, and **Streamlit**.

Three specialized AI agents collaborate in a stateful graph to turn any research topic into a structured markdown report — automatically.

---

## Architecture

```
User prompt
    │
    ▼
Research Agent  ──(insufficient?)──► Research Agent (retry, max 2×)
    │
    │ (sufficient results)
    ▼
Summarizer Agent
    │
    ▼
Report Writer Agent
    │
    ▼
Structured Markdown Report
```

| Agent | Responsibility |
|---|---|
| **Research Agent** | Queries Wikipedia and returns raw article content |
| **Summarizer Agent** | Distils raw results into bullet-point key findings |
| **Report Writer Agent** | Compiles a structured report: Overview, Key Findings, Sources |

The graph uses LangGraph's `MemorySaver` checkpointer to persist state across every step, and a conditional router that loops back to the Research Agent if results are too thin (up to 2 retries).

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd multi-agent-research-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key (get one free at <https://console.groq.com>):

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

> Wikipedia requires **no API key**.

---

## Running the application

### Start the FastAPI backend

```bash
uvicorn api.app:app --reload
```

The API will be available at:
- **REST endpoint:** `http://localhost:8000/research`
- **Swagger UI docs:** `http://localhost:8000/docs`
- **ReDoc docs:** `http://localhost:8000/redoc`

### Start the Streamlit frontend (separate terminal)

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser, enter a topic, and click **Run Research**.

---

## API usage

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Latest breakthroughs in fusion energy"}'
```

Response:

```json
{
  "topic": "Latest breakthroughs in fusion energy",
  "report": "## Overview\n..."
}
```

---

## Running tests

```bash
pytest tests/ -v
```

The smoke tests validate imports, graph topology, and request/response models without making real API calls.

---

## Project structure

```
multi-agent-research-assistant/
├── agents/
│   ├── research_agent.py       # Wikipedia search
│   ├── summarizer_agent.py     # LLM-powered summarization
│   └── report_writer_agent.py  # Markdown report generation
├── graph/
│   ├── state.py                # Shared TypedDict state schema
│   └── builder.py              # LangGraph StateGraph + conditional routing
├── api/
│   ├── app.py                  # FastAPI app factory
│   └── routes.py               # POST /research endpoint
├── frontend/
│   └── app.py                  # Streamlit UI
├── tests/
│   └── test_smoke.py           # Import and schema smoke tests
├── .env.example                # Template — copy to .env
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for Llama 3.1 inference |
