import os 
from groq import Groq
from dotenv import load_dotenv
from services.whether_service import get_weather_by_city

load_dotenv() #loads the environment variables

api_key=os.getenv("MY_OPENAI_API_KEY")

client = Groq(api_key=api_key)

async def get_suggestion(city:str):
    weather=await get_weather_by_city(city)

    if weather is None:
        return None
    
    city=weather["city"]
    temperature=weather["temperature"]
    windspeed=weather["windspeed"]
    condition=weather["condition"]

    prompt=f"""
    The current weather in {city} is:

    Temperature: {temperature}°C
    Condition: {condition}
    Wind Speed: {windspeed} km/h

    Give a short practical suggestion for the user in one sentence.
    """

    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )

    suggestion=response.choices[0].message.content

    return{
        "city": city,
        "temperature": temperature,
        "condition": condition,
        "windspeed": windspeed,
        "suggestion": suggestion
    }

    

  