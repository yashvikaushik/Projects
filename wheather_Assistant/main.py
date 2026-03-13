from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI,HTTPException,Form
from services.whether_service import fetch_coordinates,fetch_weather,get_weather_by_city
from services.groq_api_service import get_suggestion
from services.whatsapp_service import send_whatsapp_message

app=FastAPI()


@app.get('/')
def intro():
    return{"mesage":'server started successfully'}


@app.get("/coordinates/{city}")
async def get_coordinates(city:str):
    result =await fetch_coordinates(city)
    if result is None:
        raise HTTPException(status_code=404,detail="City not found")
    return result

@app.get("/weather/{latitude}&{longitude}")
async def get_weather(latitude:float,longitude:float):
    result =await fetch_weather(latitude,longitude)
    return result

@app.get("/temperature/{city}")
async def get_temp(city:str):
    result=await get_weather_by_city(city)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="City Not found"
        )
    
    return result

@app.get("/suggestion/{city}")
async def get_suggections(city):
    city=city.strip()
    result = await get_suggestion(city)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="City not found"
        )
    
    return result


@app.get("/send-weather/{city}/{phone}")
async def send_weather(city: str, phone: str):

    result = await get_suggestion(city)

    message = f"""
Weather Update 🌤

City: {result['city']}
Temperature: {result['temperature']}°C
Condition: {result['condition']}
Wind Speed: {result['windspeed']} km/h

Suggestion:
{result['suggestion']}
"""

    send_whatsapp_message(phone, message)

    return {"message": "Weather sent on WhatsApp"}


@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...)
):

    message = Body.lower()

    if "weather" in message:

        city = message.replace("weather", "").strip()

        result = await get_suggestion(city)

        reply = f"""
Weather Update 🌤

City: {result['city']}
Temperature: {result['temperature']}°C
Condition: {result['condition']}
Wind Speed: {result['windspeed']} km/h

Suggestion 
{result['suggestion']}
"""

        phone = From.replace("whatsapp:", "")

        send_whatsapp_message(phone, reply)

    return "ok"