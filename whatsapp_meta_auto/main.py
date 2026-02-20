from fastapi import FastAPI
from schemas import Message
from services import whatsapp_messages
app=FastAPI()


@app.get('/')
def home():
    return "server runnning successfully"

@app.post('/send')
def send_messages(payload:Message):
    return whatsapp_messages(payload.msg)
    
    