from fastapi import FastAPI,HTTPException
from services.whether_service import fetch_coordinates,fetch_weather,get_weather_by_city


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

