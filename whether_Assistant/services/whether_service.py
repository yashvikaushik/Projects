from pydantic import BaseModel
import httpx 
import json
from utils.weather_codes import get_weather_description

async def fetch_coordinates(city:str):
    url ="https://geocoding-api.open-meteo.com/v1/search?name="+city

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url)

    data=response.json()

    if "results" not in data or data["results"] is None:
        return None
    
    location=data["results"][0] #we take the first location
    return {
        "city": location["name"],
        "latitude": location["latitude"],
        "longitude": location["longitude"]
    }


async def fetch_weather(latitude:float,longitude:float):
    url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={latitude}&longitude={longitude}&current_weather=true"
)
    print("Weather URL:", repr(url))

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url)
    response.raise_for_status()

    data=response.json()
    
    weather = data["current_weather"]

    temperature = weather["temperature"]
    windspeed=weather["windspeed"]
    weathercode=weather["weathercode"]
    condition=get_weather_description(weathercode)

    return {
        "temperature": temperature,
        "condition":condition,
        "windspeed":windspeed
    }

async def get_weather_by_city(city:str):
    coordinates=await fetch_coordinates(city)
    if coordinates is None:
        return None
    latitude=coordinates["latitude"]
    longitude=coordinates["longitude"]

    weather=await fetch_weather(latitude,longitude)
    return {
        "city": coordinates["city"],
        "temperature": weather["temperature"],
        "windspeed":weather["windspeed"],
        "condition":weather["condition"]

    }
