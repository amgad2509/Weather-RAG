# src/tools/dummy_weather.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool


class DummyWeatherInput(BaseModel):
    """Input schema for dummy weather tool."""
    location: str = Field(..., description="City/Country name (e.g., 'Cairo', 'Doha, Qatar')")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference",
    )
    include_forecast: bool = Field(
        default=False,
        description="Include a simple 5-day forecast (dummy).",
    )


@tool("dummy_weather", args_schema=DummyWeatherInput)
def dummy_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """
    Dummy weather tool (no external APIs). Returns JSON string.
    """
    loc = " ".join((location or "").strip().split())
    if not loc:
        return json.dumps(
            {"error": "missing_location", "message": "Please provide a valid location (country/city)."},
            ensure_ascii=False,
        )

    # synthetic base temp
    temp_c = 25
    temp = temp_c if units == "celsius" else round((temp_c * 9 / 5) + 32)

    payload = {
        "provider": "dummy",
        "location": loc,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "units": units,
        "condition": "clear sky",
        "temperature": temp,
        "feels_like": temp + (1 if units == "celsius" else 2),
        "wind_mps": 3.2,
        "humidity_percent": 45,
        "precip_mm": 0.0,
    }

    if include_forecast:
        payload["forecast_5d"] = [
            {"day": 1, "condition": "sunny"},
            {"day": 2, "condition": "sunny"},
            {"day": 3, "condition": "partly cloudy"},
            {"day": 4, "condition": "sunny"},
            {"day": 5, "condition": "sunny"},
        ]

    return json.dumps(payload, ensure_ascii=False)

