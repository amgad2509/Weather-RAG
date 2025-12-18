from pydantic import BaseModel, Field


class WeatherQueryInput(BaseModel):
    """Input schema for weather_query tool."""
    location: str = Field(
        ...,
        description="City/Country name to fetch weather for (e.g., 'Doha', 'Cairo, Egypt').",
        min_length=1,
    )
