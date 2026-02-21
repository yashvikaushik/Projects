from fastapi import FastAPI
from schemas import Message
from services import whatsapp_messages
from scheduler import start_scheduler

app=FastAPI()


@app.get('/')
def home():
    return "server runnning successfully"

@app.post('/send')
def send_messages(payload:Message):
    return whatsapp_messages(payload.msg)

@app.on_event("startup")
def start_background_tasks():
    start_scheduler()
    
@app.on_event("startup")
def start_background_tasks():
    print("Startup event triggered")
    start_scheduler()
