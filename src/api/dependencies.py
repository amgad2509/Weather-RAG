from __future__ import annotations

from fastapi import Request

from src.agent.weather_agent import WeatherActivityClothingAgent


def get_agent(request: Request) -> WeatherActivityClothingAgent:
    """
    Lazily attach a single WeatherActivityClothingAgent instance to the FastAPI app state.
    """
    agent = getattr(request.app.state, "weather_agent", None)
    if agent is None:
        request.app.state.weather_agent = WeatherActivityClothingAgent()
        agent = request.app.state.weather_agent
    return agent
