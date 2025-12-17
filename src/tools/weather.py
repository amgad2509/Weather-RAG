from langchain_core.tools import tool

_BAD_LOCATIONS = {"?", "unknown", "n/a", "na", "none", "null", ""}


def make_weather_query_tool(weather_wrapper):
    @tool
    def weather_query(location: str) -> str:
        """
        Fetches real-time weather data for a specified location using OpenWeatherMap API.
        """
        loc = (location or "").strip()
        if loc.lower() in _BAD_LOCATIONS:
            return "ERROR: invalid location. Ask the user: Which location (country/city)?"
        return weather_wrapper.run(loc)

    return weather_query
