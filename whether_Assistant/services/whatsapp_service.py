import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

client = Client(account_sid, auth_token)

print("SID:", account_sid)
print("TOKEN:", auth_token)


def send_whatsapp_message(to_number: str, message: str):
    client.messages.create(
        from_=twilio_number,
        body=message,
        to=f"whatsapp:{to_number}"
    )
