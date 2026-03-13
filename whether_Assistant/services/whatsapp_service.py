import os
from twilio.rest import Client
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)



account_sid = os.getenv("ACCOUNT_SID")
auth_token = os.getenv("AUTH_TOCKEN")
twilio_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

client = Client(account_sid, auth_token)

def send_whatsapp_message(to_number: str, message: str):

    client.messages.create(
        from_=f"whatsapp:{twilio_number}",
        body=message,
        to=f"whatsapp:{to_number}"
    )