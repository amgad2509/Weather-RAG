from langchain_core.tools import tool
from src.schema import WeatherQueryInput

_BAD_LOCATIONS = {"?", "unknown", "n/a", "na", "none", "null", ""}


def make_weather_query_tool(weather_wrapper):
    @tool(args_schema=WeatherQueryInput)
    def weather_query(location: str) -> str:
        """
        Fetches real-time weather data for a specified location using OpenWeatherMap API.

        Provides detailed weather information including temperature, humidity, wind speed,
        and overall weather conditions for the given country or city.

        Args:
            location (str): The name of the location to get weather information for.

        Returns:
            str: A descriptive weather report string with current meteorological data.
        """
        loc = (location or "").strip()
        if loc.lower() in _BAD_LOCATIONS:
            return "ERROR: invalid location. Ask the user: Which location (country/city)?"
        return weather_wrapper.run(loc)

    return weather_query
