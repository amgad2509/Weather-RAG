from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.weather_agent import WeatherActivityClothingAgent
from src.api.routes.base_route import base_router
from src.api.tracing_logger import setup_tracing_logger

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create heavy resources once on startup (agent + tracing logger).
    """
    try:
        app.state.weather_agent = WeatherActivityClothingAgent()
        app.state.tracing = setup_tracing_logger(os.getenv("TRACING_LOG_PATH", "tracing.log"))
        logger.info("WeatherActivityClothingAgent initialized successfully.")
    except Exception as exc:
        logger.exception("Failed to initialize WeatherActivityClothingAgent.")
        raise exc

    yield

    # No explicit teardown required currently.


def create_app() -> FastAPI:
    app = FastAPI(
        title="Weather Chatbot RAG API",
        version=os.getenv("APP_VERSION", "0.1.0"),
        lifespan=lifespan,
    )

    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allow_origins.split(",")] if allow_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(base_router)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "agent_initialized": hasattr(app.state, "weather_agent"),
        }

    return app


app = create_app()

__all__ = ["create_app", "app"]
