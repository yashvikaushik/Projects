# 🌦️ AI WhatsApp Weather Assistant  
### FastAPI + Twilio + Groq LLM + Weather APIs

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Twilio](https://img.shields.io/badge/Twilio-WhatsApp-red)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

An **AI-powered WhatsApp Weather Assistant** built using **FastAPI, Twilio, Groq LLM, and Weather APIs**.

Users can send a message on WhatsApp asking about weather conditions, and the system intelligently responds with weather insights and suggestions.

---

# 🚀 Features

✅ Receive WhatsApp messages via **Twilio Webhooks**  
✅ Process user queries using **Groq LLM**  
✅ Fetch real-time weather data using **Weather APIs**  
✅ Automatic WhatsApp responses  
✅ FastAPI backend architecture  
✅ Local development using **ngrok**

---

# 🏗️ Architecture

```
User WhatsApp Message
        │
        ▼
Twilio WhatsApp Sandbox
        │
        ▼
Webhook Request
        │
        ▼
FastAPI Backend
        │
        ├── Weather API (Geo / Meteo)
        │
        └── Groq LLM Processing
        │
        ▼
AI Generated Weather Suggestion
        │
        ▼
WhatsApp Response to User
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|-------------|-------------|
| Python | Programming language |
| FastAPI | Backend framework |
| Twilio | WhatsApp messaging API |
| Groq LLM | Natural language processing |
| Geo / Meteo APIs | Weather data |
| ngrok | Public tunnel for local server |
| Uvicorn | ASGI server |

---

# 📂 Project Structure

```
whatsapp-weather-ai
│
├── main.py
├── weather_service.py
├── llm_service.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/whatsapp-weather-ai.git
cd whatsapp-weather-ai
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment

Mac / Linux

```bash
source venv/bin/activate
```

Windows

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
requests
groq
python-multipart
```

---

# ▶️ Running the Server

Start the FastAPI server

```bash
uvicorn main:app --reload
```

Server will run at

```
http://127.0.0.1:8000
```

---

# 🌐 Expose Local Server

Use **ngrok** so Twilio can access your webhook.

```bash
ngrok http 8000
```

Example

```
https://abcd1234.ngrok.io
```

---

# 📲 Twilio WhatsApp Setup

1. Go to **Twilio Console**
2. Navigate to **Messaging → WhatsApp Sandbox**
3. Join sandbox using provided WhatsApp code
4. Set webhook

Example

```
https://abcd1234.ngrok.io/webhook
```

Method

```
POST
```

---

# 📡 API Endpoint

## POST `/webhook`

Receives WhatsApp messages and processes weather queries.

Example:

```python
@app.post("/webhook")
async def whatsapp_bot(Body: str = Form(...)):

    weather = get_weather_data(Body)

    suggestion = groq_llm(weather)

    return send_whatsapp_message(suggestion)
```

---

# 🧠 How It Works

1️⃣ User sends WhatsApp message (e.g. *"Weather in Delhi"*)  
2️⃣ Twilio sends webhook request to FastAPI  
3️⃣ Backend fetches weather data from API  
4️⃣ Groq LLM generates human-like weather suggestion  
5️⃣ Response is sent back via WhatsApp  

---

# 🧪 Example Query

User message:

```
Weather in Delhi today
```

Bot response:

```
The weather in Delhi today is 32°C with clear skies.
It’s a great day for outdoor activities but stay hydrated due to the heat.
```

---

# 🔐 Environment Variables

```
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_WHATSAPP_NUMBER
GROQ_API_KEY
WEATHER_API_KEY
```

---

# 🚀 Future Improvements

- Multi-city weather queries
- AI travel recommendations
- Weather alerts
- WhatsApp chatbot with memory
- AI clothing suggestions based on weather

---

# 📜 License

MIT License