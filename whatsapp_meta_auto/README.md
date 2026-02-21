                      📱 WhatsApp Automated Message Scheduler
                      
An automated WhatsApp messaging system built using FastAPI, APScheduler, and Meta WhatsApp Cloud API.
This project sends scheduled WhatsApp messages (for example, a daily Good Morning message) using a production-ready System User access token.

🚀 Features
-Send custom WhatsApp messages via REST API
-Schedule automatic daily messages using cron
-Secure authentication using Meta System User token
-Clean backend architecture (API layer + service layer + scheduler)
-Environment-based configuration
-Background task execution with APScheduler

🧠 How It Works
->FastAPI server starts
->Startup event triggers background scheduler
->Scheduler runs a cron or interval job
->Service layer sends request to Meta WhatsApp Cloud API
->Message is delivered to the recipient

📂 Project Structure
whatsapp_meta_auto/
│
├── main.py          # FastAPI app & startup hook
├── services.py      # WhatsApp messaging logic
├── scheduler.py     # Background scheduler setup
├── schemas.py       # Pydantic request models
├── config.py        # Environment variable loader
├── .env             # Secrets (not committed)
├── requirements.txt
└── README.md

🔐 Authentication
This project uses a System User Access Token generated from Meta Business Manager.
Required permissions:
whatsapp_business_messaging
whatsapp_business_management