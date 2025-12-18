# src/api/main.py
from __future__ import annotations

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.base_route import base_router
from src.agent.weather_agent import WeatherActivityClothingAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather Chatbot RAG API",
    version=os.getenv("APP_VERSION", "0.1.0"),
)

# CORS
allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allow_origins.split(",")] if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(base_router)

@app.on_event("startup")
def startup_init_agent():
    """
    Create heavy resources once on startup.
    """
    try:
        app.state.weather_agent = WeatherActivityClothingAgent()
        logger.info("WeatherActivityClothingAgent initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize WeatherActivityClothingAgent.")
        # هنا الأفضل نفشل التشغيل بدل ما يشتغل API بدون Agent
        raise

@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_initialized": hasattr(app.state, "weather_agent"),
    }
