# Project: Multi-Agent Research Assistant

## Git Workflow
- Branch: dev
- Conventional Commits format (feat:, fix:, chore:, refactor:)
- Never commit .env

## Stack
- Python, LangGraph, LangChain, ChatGroq (Llama 3.1), DuckDuckGo, FastAPI, Streamlit

## Structure
- agents/    — one file per agent (research, summarizer, report_writer)
- graph/     — LangGraph state definition and graph builder
- api/       — FastAPI app and routes
- frontend/  — Streamlit app
- tests/     — basic smoke tests

## Rules
- All secrets in .env only (GROQ_API_KEY)
- Use python-dotenv to load environment variables
- Inline comments on all agents and graph edges
- One responsibility per file
- Auto-generated API docs at /docs
