# src/api/routes/base_route.py
from fastapi import APIRouter
from src.api.routes.chat import ai_agent_router

base_router = APIRouter(prefix="/api/v1")

base_router.include_router(ai_agent_router)
