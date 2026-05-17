"""
FastAPI application factory.

Loads environment variables from .env, mounts the research router, and
exposes auto-generated docs at /docs (Swagger UI) and /redoc (ReDoc).
"""

from dotenv import load_dotenv

# Load GROQ_API_KEY (and any other vars) from .env before anything else runs.
load_dotenv()

from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description=(
        "A LangGraph-powered research pipeline with three specialized agents: "
        "Research, Summarizer, and Report Writer."
    ),
    version="1.0.0",
)

# Mount all routes defined in routes.py under the root path.
app.include_router(router)


@app.get("/", summary="Health check")
async def root() -> dict:
    """Simple liveness probe — returns service status."""
    return {"status": "ok", "service": "multi-agent-research-assistant"}
