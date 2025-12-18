# src/api/main.py
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.base_route import base_router
from src.agent.weather_agent import WeatherActivityClothingAgent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create heavy resources once on startup (agent + vectorstore + etc).
    """
    try:
        app.state.weather_agent = WeatherActivityClothingAgent()
        logger.info("WeatherActivityClothingAgent initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize WeatherActivityClothingAgent.")
        # fail fast: API should not run half-broken
        raise e

    yield

    # No explicit teardown required currently


def create_app() -> FastAPI:
    app = FastAPI(
        title="Weather Chatbot RAG API",
        version=os.getenv("APP_VERSION", "0.1.0"),
        lifespan=lifespan,
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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
