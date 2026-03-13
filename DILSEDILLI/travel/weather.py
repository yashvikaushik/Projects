"""
weather.py
----------
Handles all weather-related functionality:
  - Fetching live weather from OpenWeatherMap API
  - Generating weather-based visit recommendations
"""

import os
import requests
from typing import Optional

# ── API Configuration ────────────────────────────────────────────────────────
# Set your key via environment variable WEATHER_API_KEY, or replace the
# fallback string below with your actual key.
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "d5817a4adf68f8fe8ada217ff2f7de4c")
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Types considered "indoor-friendly" for rainy-day recommendations
INDOOR_PLACE_TYPES = [
    "Museum", "Temple", "Monument", "Observatory",
    "Art Gallery", "Science Centre", "Historical", "Religious"
]


def get_weather(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch real-time weather for a given latitude/longitude.

    Args:
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        Dictionary with weather data, or None if the request fails.
        Keys: temperature, feels_like, humidity, wind_speed, rain_1h, description
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric",   # Celsius; change to 'imperial' for Fahrenheit
    }

    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            print(f"⚠️  Weather API error: {data.get('message', 'Unknown error')}")
            return None

        return {
            "temperature": data["main"]["temp"],
            "feels_like":  data["main"]["feels_like"],
            "humidity":    data["main"]["humidity"],
            "wind_speed":  data["wind"]["speed"],
            "rain_1h":     data.get("rain", {}).get("1h", 0),
            "description": data["weather"][0]["description"].capitalize(),
        }

    except requests.exceptions.RequestException as e:
        print(f"⚠️  Could not connect to weather API: {e}")
        return None


def is_raining(weather: dict) -> bool:
    """
    Determine whether it is currently raining based on weather data.

    Args:
        weather: Weather dict returned by get_weather()

    Returns:
        True if it is raining / drizzling / stormy, False otherwise
    """
    if not weather:
        return False

    description = weather.get("description", "").lower()
    rain_mm = weather.get("rain_1h", 0)

    return (
        rain_mm > 0
        or "rain" in description
        or "drizzle" in description
        or "thunderstorm" in description
    )


def get_weather_recommendation(weather: Optional[dict]) -> str:
    """
    Generate a human-readable weather recommendation string.

    Args:
        weather: Weather dict returned by get_weather(), or None

    Returns:
        A recommendation string suitable for inclusion in the API response
    """
    if not weather:
        return "Weather data unavailable."

    if is_raining(weather):
        return (
            "It's currently raining. Consider visiting indoor attractions such as "
            "museums, temples, or monuments. Carry an umbrella if heading out."
        )

    return (
        f"Weather looks great ({weather['description']}) — "
        "perfect time to explore both indoor and outdoor places!"
    )
