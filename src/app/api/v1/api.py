"""API v1 router aggregator."""

from fastapi import APIRouter

from src.app.api.v1.endpoints import agent, health, llm, orchestration

api_v1_router = APIRouter()
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(agent.router, prefix="/agent", tags=["Agent"])
api_v1_router.include_router(llm.router, prefix="/llm", tags=["LLM"])
api_v1_router.include_router(orchestration.router, prefix="/orchestration", tags=["Orchestration"])
