# 📞 FastAPI + Twilio Voice Automation API

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Twilio](https://img.shields.io/badge/Twilio-Voice-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A simple **Voice Automation API** built using **FastAPI and Twilio**.  
This application receives **incoming phone calls** via Twilio and responds with automated voice instructions using **TwiML**.

The project uses **ngrok** to expose the local FastAPI server to the internet so that Twilio can send webhook requests.

---

# 🚀 Features

✅ Handle incoming calls using Twilio  
✅ Generate automated voice responses using TwiML  
✅ Webhook handling with FastAPI  
✅ Local development with ngrok  
✅ Easy deployment to Railway / Render / AWS  

---

# 🏗️ Architecture

```
User Phone Call
       │
       ▼
Twilio Phone Number
       │
       ▼
Twilio Webhook
       │
       ▼
Public URL (ngrok / deployed server)
       │
       ▼
FastAPI Application
       │
       ▼
TwiML Voice Response
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|-------------|-------------|
| Python | Programming language |
| FastAPI | Backend API framework |
| Twilio | Voice communication API |
| ngrok | Expose local server |
| Uvicorn | ASGI server |

---

# 📂 Project Structure

```
twilio-fastapi-voice-bot
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/twilio-fastapi-voice-bot.git
cd twilio-fastapi-voice-bot
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it

### Mac / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies

```
fastapi
uvicorn
twilio
python-multipart
```

---

# ▶️ Running the Server

Start the FastAPI server

```bash
uvicorn main:app --reload
```

Server will run on

```
http://127.0.0.1:8000
```

---

# 🌐 Expose Local Server with ngrok

Twilio requires a **public URL** to send webhook requests.

Run:

```bash
ngrok http 8000
```

Example:

```
https://1234abcd.ngrok.io
```

---

# ☎️ Configure Twilio Webhook

1. Go to **Twilio Console**
2. Navigate to **Phone Numbers**
3. Select your Twilio number
4. Set **Voice Webhook**

Example:

```
https://1234abcd.ngrok.io/voice
```

Method

```
POST
```

Now whenever someone calls the Twilio number, the request will hit your **FastAPI endpoint**.

---

# 📡 API Endpoint

## POST `/voice`

Handles incoming calls and returns a TwiML voice response.

### Example Implementation

```python
from fastapi import FastAPI
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

app = FastAPI()

@app.post("/voice")
async def voice():

    response = VoiceResponse()

    response.say(
        "Hello! This call is handled using FastAPI and Twilio."
    )

    return Response(
        content=str(response),
        media_type="application/xml"
    )
```

---

# 🧪 Testing

1️⃣ Start FastAPI server  
2️⃣ Run ngrok  
3️⃣ Configure Twilio webhook  
4️⃣ Call your Twilio phone number  

You should hear the automated voice message.

---

# 🚀 Deployment

You can deploy this API on:

- Railway
- Render
- AWS EC2
- Docker

Once deployed, **ngrok will not be required** because the server will already have a public URL.

---

# 🔐 Environment Variables

For security, store credentials using environment variables.

Example:

```
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER
```

---

# 📚 Learning Outcomes

This project demonstrates:

- FastAPI backend development
- Webhook integration
- Twilio Voice API
- ngrok tunneling
- Building voice automation systems

---

# 📜 License

MIT License

---

# ⭐ If you found this useful

Give the repository a **star ⭐ on GitHub**.