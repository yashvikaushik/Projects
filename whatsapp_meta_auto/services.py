import requests
from config import ACCESS_TOKEN,PHONE_NUMBER_ID,RECIPIENT_PHONE 

def whatsapp_messages(msg:str):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        
    "messaging_product": "whatsapp",
    "to": RECIPIENT_PHONE,
    "type": "text",
    "text":{"body":msg}
    }  
    print("Payload being sent to Meta:", data)

    response=requests.post(url,headers=headers,json=data)

    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    return response.json()
